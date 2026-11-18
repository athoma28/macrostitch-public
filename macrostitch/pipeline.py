"""A disk-backed, stage-cached focus-stack pipeline for static macro video."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from fractions import Fraction
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import cv2
import numpy as np
from .imaging import (focus_map, gray, roi_mask, register, scale_transform,
    select_coverage, to_linear_bt709, to_srgb16, laplacian_pyramid, reconstruct, common_crop)
from .cache_io import image_record, atomic_image_write, stamp_image
from .storage import (DEFAULT_CACHE_MAX_BYTES, cache_retention, clear_frame_exports,
                      pipeline_workspace_guard, validate_limit)

VERSION = "3.3.0"
# Registration is unchanged; preserve already-verified geometry work across fusion revisions.
GEOMETRY_VERSION = "1.0.2"


@dataclass
class Config:
    sample_fps: float = 10.
    proxy_width: int = 960
    max_frames: int = 48
    coverage: float = .985
    roi: tuple[float,float,float,float] | None = None
    pyramid_levels: int = 6
    threads: int = 4
    subject_only: bool = True
    subject_polygon: tuple[tuple[float,float],...] | None = None
    fusion_mode: str = "pyramid"
    selection_mode: str = "proxy"
    spatial_blend_px: float = 1.
    spatial_smoothness: float = .12
    local_alignment_regions: tuple[dict,...] | None = None
    required_frames: tuple[dict,...] | None = None
    background_mode: str = "reference"
    background_opacity: float | None = None
    background_margin_px: float = 0.
    object_masks: tuple[dict,...] | None = None
    canvas_mode: str = "common"
    focus_regions: tuple[dict,...] | None = None
    score_normalization_roi: tuple[float,float,float,float] | None = None
    region_blend_px: float = 3.
    matte_feather_px: float = 1.5
    matte_erode_px: float = 1.

    def validate(self):
        from .compositing import normalize_controls, controls_active
        normalize_controls(self)
        # Canonicalize equivalent Python/JSON configurations before hashing them.
        self.sample_fps=float(self.sample_fps)
        self.coverage=float(self.coverage)
        for name in ("proxy_width","max_frames","pyramid_levels","threads"):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or int(value)!=value:
                raise ValueError(f"{name} must be an integer")
            setattr(self,name,int(value))
        for name,lo,hi in [("region_blend_px",0,32),("matte_feather_px",.1,8),("matte_erode_px",0,8),("spatial_blend_px",0,4),("spatial_smoothness",0,1)]:
            value=float(getattr(self,name)); setattr(self,name,value)
            if not math.isfinite(value) or not lo<=value<=hi: raise ValueError(f"Invalid {name}")
        if self.fusion_mode not in ("pyramid","regional","spatial") or self.background_mode not in ("reference","transparent"):
            raise ValueError("Invalid fusion_mode or background_mode")
        if self.selection_mode not in ("proxy","native"):
            raise ValueError("Invalid selection_mode")
        if self.canvas_mode not in ("common","reference"):
            raise ValueError("Invalid canvas_mode")
        if self.local_alignment_regions is not None:
            from .local_alignment import normalize_local_regions
            self.local_alignment_regions=normalize_local_regions(self.local_alignment_regions)
            if self.fusion_mode!='spatial' or self.selection_mode!='native':
                raise ValueError('Local alignment requires native selection and spatial fusion')
        if self.required_frames is not None:
            if self.selection_mode!='native':
                raise ValueError('required_frames requires native selection')
            if not isinstance(self.required_frames,(tuple,list)) or not 1<=len(self.required_frames)<=512:
                raise ValueError('required_frames requires 1..512 source/frame dictionaries')
            anchors=[]
            for anchor in self.required_frames:
                if (not isinstance(anchor,dict) or set(anchor)!={'source','frame'}
                        or any(type(anchor[k]) is not int or anchor[k]<0 for k in ('source','frame'))):
                    raise ValueError('required_frames requires nonnegative integer source and frame indices')
                anchors.append((anchor['source'],anchor['frame']))
            if len(set(anchors))!=len(anchors):
                raise ValueError('required_frames must not contain duplicates')
            if len(anchors)>self.max_frames:
                raise ValueError('required_frames exceeds the frame budget')
            self.required_frames=tuple({'source':source,'frame':frame} for source,frame in sorted(anchors))
        def checked_roi(roi):
            value=tuple(float(v) for v in roi)
            if len(value)!=4 or not all(math.isfinite(v) for v in value): raise ValueError("ROI requires four finite coordinates")
            a,b,c,d=value
            if not 0<=a<c<=1 or not 0<=b<d<=1: raise ValueError("ROI must be normalized ordered x0,y0,x1,y1")
            return value
        if self.score_normalization_roi is not None:
            self.score_normalization_roi=checked_roi(self.score_normalization_roi)
        if self.fusion_mode=="regional" and self.focus_regions is None:
            raise ValueError("Regional fusion requires focus_regions")
        if self.background_mode=="transparent" and self.subject_polygon is None and not controls_active(self):
            raise ValueError("Transparent background requires subject_polygon")
        if self.focus_regions is not None:
            if not 1<=len(self.focus_regions)<=32: raise ValueError("Requires 1..32 focus regions")
            normalized=[]; previous=0.
            for region in self.focus_regions:
                if not isinstance(region,dict) or set(region)-{"y","roi","tone_roi","tone_reference"}:
                    raise ValueError("Invalid focus region keys")
                bounds=tuple(float(v) for v in region["y"])
                if len(bounds)!=2 or bounds[0]!=previous or not 0<=bounds[0]<bounds[1]<=1:
                    raise ValueError("Focus regions must consecutively cover [0,1]")
                row={"y":bounds,"roi":checked_roi(region["roi"])}; previous=bounds[1]
                if region.get("tone_roi") is not None: row["tone_roi"]=checked_roi(region["tone_roi"])
                anchor=region.get("tone_reference")
                if anchor is not None:
                    if type(anchor) is not int or not 0<=anchor<len(self.focus_regions): raise ValueError("Invalid tone_reference")
                    row["tone_reference"]=anchor
                normalized.append(row)
            if previous!=1.: raise ValueError("Focus regions must consecutively cover [0,1]")
            self.focus_regions=tuple(normalized)
        if self.roi is not None: self.roi=tuple(float(v) for v in self.roi)
        if self.subject_polygon is not None:
            self.subject_polygon=tuple(tuple(float(v) for v in p) for p in self.subject_polygon)
        if self.subject_polygon is not None:
            if not 3<=len(self.subject_polygon)<=128 or any(len(p)!=2 or not all(math.isfinite(v) and 0<=v<=1 for v in p) for p in self.subject_polygon):
                raise ValueError("subject_polygon requires 3..128 normalized x,y points")
        if not 0 < self.sample_fps <= 120 or not 64 <= self.proxy_width <= 4096:
            raise ValueError("Invalid sample_fps or proxy_width")
        if not 2 <= self.max_frames <= 512 or not .5 <= self.coverage <= 1:
            raise ValueError("Invalid max_frames or coverage")
        if not 1 <= self.pyramid_levels <= 12 or not 1 <= self.threads <= 64:
            raise ValueError("Invalid pyramid_levels or threads")
        if self.roi is not None:
            if len(self.roi)!=4 or not all(math.isfinite(v) for v in self.roi):
                raise ValueError("ROI requires four finite coordinates")
            a,b,c,d=self.roi
            if not 0<=a<c<=1 or not 0<=b<d<=1:
                raise ValueError("ROI must be normalized ordered x0,y0,x1,y1")


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def key(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:24]


def save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False))
    os.replace(temporary,path)


def load_json(path):
    """A damaged completion record is a cache miss, not a permanently failed job."""
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else None
    except (OSError,ValueError,TypeError):
        return None


def artifact_names(config):
    names=["stack.tiff","stack.png","preview.jpg","comparison.jpg","source_index.png","confidence.png","selection.json","events.json"]
    if config.fusion_mode in ("regional","spatial"): names.append("fusion.json")
    if config.fusion_mode=="spatial": names.append("subject-detail.png")
    if config.selection_mode=="native": names.append("native-analysis.json")
    if config.local_alignment_regions: names.append('local-alignment.json')
    from .compositing import controls_active
    if controls_active(config):
        names.extend(["subject.png","matte.png","reference.png","compositing.json"])
    elif config.background_mode=="transparent": names.extend(["subject.png","matte.png"])
    return names


def final_cache_valid(previous,output,identifier,config):
    if not isinstance(previous,dict): return False
    try:
        if not isinstance(previous.get('artifacts'),dict) or not isinstance(previous.get('artifact_sha256'),dict):return False
        prior=Config(**previous["config"]);prior.validate()
        previous_key=key([previous["version"],[s["sha256"] for s in previous["sources"]],asdict(prior)])
        expected=set(artifact_names(config))
        if previous_key!=identifier or set(previous["artifacts"])!=expected|{"report.json"} or set(previous["artifact_sha256"])!=expected:
            return False
        for name in ("result","coverage","selected_frames","candidates","source_frames","elapsed_seconds"):
            if name not in previous:return False
        return all((output/name).is_file() and digest(output/name)==checksum for name,checksum in previous["artifact_sha256"].items())
    except (KeyError,TypeError,ValueError,OSError):
        return False


def atomic_save_array(path,array):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npy',delete=False) as stream:
        temporary=Path(stream.name)
        try:
            np.save(stream,array,allow_pickle=False)
            stream.flush()
            os.replace(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield


def command(args,log):
    with open(log,"ab") as f:
        result=subprocess.run(args,stdout=f,stderr=f)
    if result.returncode:
        tail=Path(log).read_text(errors="replace")[-3000:]
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {tail}")


def probe(path):
    data=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
        "-show_streams","-show_format","-of","json",str(path)]))
    s=data["streams"][0]
    if s.get("color_transfer") not in (None,"unknown","bt709"):
        raise ValueError(f"Unsupported transfer {s.get('color_transfer')}; convert HDR/log footage to BT.709 first")
    if s.get("color_space") not in (None,"unknown","bt709"):
        raise ValueError(f"Unsupported matrix {s.get('color_space')}; convert to BT.709 first")
    fps=float(Fraction(s["avg_frame_rate"]))
    if fps<=0:
        raise ValueError("Cannot determine video frame rate")
    return {"path":str(path),"sha256":digest(path),"width":s["width"],"height":s["height"],
            "fps":fps,"frames":int(s.get("nb_frames",0)),"duration":float(data["format"]["duration"]),
            "pixel_format":s["pix_fmt"],"color_transfer":s.get("color_transfer","assumed bt709"),
            "color_space":s.get("color_space","assumed bt709")}


def proxies(source,cache,config,emit):
    step=max(1,round(source["fps"]/config.sample_fps))
    width=min(config.proxy_width,source["width"])
    location=cache/"proxies"/key([source["sha256"],step,width,"rgb-proxy-v1"])
    done=location/"complete.json"
    proxy_shape=(max(2,round(source['height']*width/source['width']/2)*2),width,3)
    def valid_proxy(path):
        # FFmpeg historically emitted either 8- or 16-bit RGB PNG by input depth.
        # OpenCV's analysis reader consistently converts both to 8-bit BGR.
        return (image_record(path,proxy_shape) is not None
                or image_record(path,proxy_shape,dtype=np.uint8) is not None)
    metadata=load_json(done)
    if metadata:
        count=metadata.get('count')
        times=metadata.get('timestamps')
        valid=(type(count) is int and count>0 and metadata.get('step')==step and metadata.get('width')==width
               and isinstance(times,list) and all(isinstance(t,(int,float)) and math.isfinite(t) for t in times)
               and metadata.get('timestamp_basis') in ('sorted packet PTS','frame index / average FPS'))
        if (valid and len(list(location.glob("p*.png")))==count
                and all(valid_proxy(location/f'p{i:06d}.png') for i in range(count))):
            emit(stage="proxies",message=f"Cached previews: {Path(source['path']).name}")
            return location,metadata
    location.mkdir(parents=True,exist_ok=True)
    done.unlink(missing_ok=True)
    for stale in location.glob('p*.png*'):stale.unlink()
    emit(stage="proxies",message=f"Decoding previews: {Path(source['path']).name}")
    command(["ffmpeg","-hide_banner","-loglevel","warning","-threads",str(config.threads),
        "-i",source["path"],"-an","-vf",f"select=not(mod(n\\,{step})),scale={width}:-2:flags=area",
        "-fps_mode","passthrough","-pix_fmt","rgb48be","-start_number","0","-compression_level","2","-y",str(location/"p%06d.png")],location/"decode.log")
    # Packet PTS sorted into display order gives exact timestamps for ordinary one-picture-per-packet codecs.
    pts=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "packet=pts_time","-of","csv=p=0",source["path"]],text=True)
    times=sorted(float(line.split(',')[0]) for line in pts.splitlines() if line and line.split(',')[0]!="N/A")
    count=len(list(location.glob("p*.png")))
    if count<1 or any(not valid_proxy(location/f'p{i:06d}.png') for i in range(count)):
        raise RuntimeError('Proxy decode produced missing or invalid frames')
    metadata={"count":count,"step":step,"width":width,"timestamps":times,
              "timestamp_basis":"sorted packet PTS" if len(times)>=count*step-step+1 else "frame index / average FPS"}
    save_json(done,metadata)
    return location,metadata


def analyze(sources,cache,config,emit):
    records=[]
    for source_index,source in enumerate(sources):
        directory,meta=proxies(source,cache,config,emit)
        source["timestamp_basis"]=meta["timestamp_basis"]
        if not source["frames"]: source["frames"]=len(meta["timestamps"])
        for i in range(meta["count"]):
            n=i*meta["step"]
            records.append({"source":source_index,"frame":n,
                "timestamp":meta["timestamps"][n] if n<len(meta["timestamps"]) else n/source["fps"],
                "proxy":str(directory/f"p{i:06d}.png")})
    align_key=key([GEOMETRY_VERSION,[s["sha256"] for s in sources],config.sample_fps,config.proxy_width,config.roi])
    directory=cache/"analysis"/align_key; directory.mkdir(parents=True,exist_ok=True)
    done=directory/"analysis.json"
    integrity_path=directory/"analysis.verified.json"
    def analysis_identity():
        return {"schema_version":1,"scores_sha256":digest(directory/"scores.npy"),
                "metadata_sha256":digest(done)}
    data=load_json(done)
    if data:
        try:
            identity=analysis_identity()
            # Legacy caches are adopted only after structural validation below;
            # once stamped, plausible finite edits must not silently change choices.
            if data.get("integrity_schema") is not None and not integrity_path.exists():
                raise ValueError("Analysis cache integrity record missing")
            if integrity_path.exists() and load_json(integrity_path)!=identity:
                raise ValueError("Analysis cache content changed")
            saved=data['records'];reference=data['reference']
            first=cv2.imread(records[0]['proxy'])
            shape=(len(records),math.ceil(first.shape[0]/16),math.ceil(first.shape[1]/16))
            scores=np.load(directory/'scores.npy',allow_pickle=False)
            valid=(type(reference) is int and 0<=reference<len(records) and len(saved)==len(records)
                   and scores.shape==shape and scores.dtype==np.float32 and np.isfinite(scores).all() and (scores>=0).all())
            for expected,row in zip(records,saved):
                valid &= all(row[name]==expected[name] for name in ('source','frame','proxy','timestamp'))
                matrix=np.asarray(row['transform'])
                valid &= matrix.shape==(2,3) and np.isfinite(matrix).all() and isinstance(row['accepted'],bool)
                valid &= math.isfinite(row['global_focus']) and math.isfinite(row['alignment']['correlation'])
            if valid and reference==int(np.argmax([row['global_focus'] for row in saved])):
                if identity!=analysis_identity():raise ValueError("Analysis cache changed during validation")
                if data.get("integrity_schema")!=1:
                    data["integrity_schema"]=1
                    save_json(done,data)
                    identity=analysis_identity()
                if load_json(integrity_path)!=identity:save_json(integrity_path,identity)
                emit(stage="analysis",message="Verified cached proxy registration and focus maps")
                return saved,scores,reference,directory
        except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError,EOFError):
            pass
    done.unlink(missing_ok=True)
    # Reference selection balances broad subject detail and a moderate breathing scale.
    quality=[]
    for r in records:
        image=cv2.imread(r["proxy"])
        fm=focus_map(image); mask=roi_mask(fm.shape,config.roi)>0
        quality.append(float(np.sqrt(fm[mask]).mean()))
    reference=int(np.argmax(quality))
    ref=cv2.imread(records[reference]["proxy"])
    grid=(math.ceil(ref.shape[1]/16),math.ceil(ref.shape[0]/16))
    scores=[]
    for i,r in enumerate(records):
        image=cv2.imread(r["proxy"])
        if i==reference:
            warp=np.eye(2,3,dtype=np.float32); metrics={"correlation":1.,"ecc_converged":True}
        else:
            warp,metrics=register(ref,image,config.roi)
        aligned=cv2.warpAffine(image,warp,(ref.shape[1],ref.shape[0]))
        valid=cv2.warpAffine(np.ones(image.shape[:2],np.uint8),warp,(ref.shape[1],ref.shape[0]))
        valid=cv2.erode(valid,np.ones((9,9),np.uint8))
        fm=focus_map(aligned)*valid
        # A low correlation signals motion/cut or incompatible viewpoint, not a usable focus plane.
        accepted=metrics["correlation"]>=.88
        scores.append(cv2.resize(fm,grid,interpolation=cv2.INTER_AREA) if accepted else np.zeros(grid[::-1],np.float32))
        r.update(transform=warp.tolist(),alignment=metrics,accepted=accepted,global_focus=quality[i])
        if i%20==0 or i==len(records)-1:
            emit(stage="analysis",completed=i+1,total=len(records),message="Registering previews and measuring spatial focus")
    scores=np.asarray(scores,dtype=np.float32)
    atomic_save_array(directory/"scores.npy",scores)
    save_json(done,{"records":records,"reference":reference,"integrity_schema":1})
    save_json(integrity_path,analysis_identity())
    return records,scores,reference,directory


def _frame_selection_expression(frames):
    """Keep FFmpeg's expression parser depth logarithmic for large selections."""
    terms=[f"eq(n\\,{frame})" for frame in frames]
    def combine(start,stop):
        if stop-start==1:return terms[start]
        middle=(start+stop)//2
        return f"({combine(start,middle)}+{combine(middle,stop)})"
    if not terms:raise ValueError('Frame selection cannot be empty')
    return combine(0,len(terms))


def extract(sources,records,cache,config,emit):
    for source_index,source in enumerate(sources):
        location=cache/"frames"/source["sha256"][:24]; location.mkdir(parents=True,exist_ok=True)
        selected=[r for r in records if r["source"]==source_index]
        missing=[]
        for r in selected:
            r["extracted"]=str(location/f"f{r['frame']:06d}.tiff")
            if image_record(r['extracted'],(source['height'],source['width'],3)) is None: missing.append(r)
        if not missing: continue
        missing.sort(key=lambda r:r["frame"])
        emit(stage="extract",message=f"Extracting {len(missing)} 16-bit frames from {Path(source['path']).name}")
        expression=_frame_selection_expression(r['frame'] for r in missing)
        # One sequential decode, never one seek/decode per selected frame.
        # Isolated attempts cannot accidentally promote leftovers after a short decode.
        with tempfile.TemporaryDirectory(prefix='pending-',dir=location) as attempt:
            temporary=Path(attempt)
            command(["ffmpeg","-hide_banner","-loglevel","warning","-threads",str(config.threads),
                "-i",source["path"],"-an","-vf",f"select={expression},format=rgb48le",
                "-fps_mode","passthrough","-start_number","0","-compression_algo","deflate",
                "-y",str(temporary/"f%06d.tiff")],location/"decode.log")
            if len(list(temporary.glob('f*.tiff')))!=len(missing):
                raise RuntimeError('Decoder did not return every requested frame')
            # Validate the whole attempt before publishing any frame.
            for i in range(len(missing)):
                if image_record(temporary/f'f{i:06d}.tiff',(source['height'],source['width'],3)) is None:
                    raise RuntimeError(f'Invalid extracted frame {i}')
            for i,r in enumerate(missing):
                path=temporary/f'f{i:06d}.tiff'
                image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
                os.replace(path,r['extracted'])
                stamp_image(r['extracted'],image)


def align_selected(records,reference,cache,config,emit):
    reference_record=records[reference]
    ref=cv2.imread(reference_record["proxy"])
    ref_large=cv2.imread(reference_record["extracted"],cv2.IMREAD_UNCHANGED)
    h,w=ref_large.shape[:2]
    reference_identity=image_record(reference_record['extracted'],(h,w,3))
    if reference_identity is None:raise RuntimeError('Invalid reference frame cache')
    common=np.ones((h,w),np.uint8)
    outputs=[]
    for i,r in enumerate(records):
        location=cache/"aligned"/key([GEOMETRY_VERSION,r["extracted"],reference_record["extracted"],config.roi,
                                     r["transform"],ref.shape[:2]])
        location.mkdir(parents=True,exist_ok=True)
        path=location/"aligned.tiff"; info=location/"transform.json"
        source_identity=image_record(r['extracted'],(h,w,3))
        if source_identity is None:raise RuntimeError('Invalid source frame cache')
        meta=load_json(info);usable=False
        if meta and image_record(path,(h,w,3)) is not None:
            try:
                full=np.asarray(meta['transform'],np.float32)
                usable=(full.shape==(2,3) and np.isfinite(full).all() and abs(np.linalg.det(full[:,:2]))>1e-8
                        and math.isfinite(meta['alignment']['correlation'])
                        and meta.get('source_sha256')==source_identity['sha256']
                        and meta.get('reference_sha256')==reference_identity['sha256'])
            except (KeyError,ValueError,TypeError):pass
        if not usable:
            image=cv2.imread(r["extracted"],cv2.IMREAD_UNCHANGED)
            # Refine at twice analysis resolution using the original, preserving one final resample.
            size=(min(1920,w),round(h*min(1920,w)/w))
            a=(cv2.resize(ref_large,size,interpolation=cv2.INTER_AREA)/256).astype(np.uint8)
            b=(cv2.resize(image,size,interpolation=cv2.INTER_AREA)/256).astype(np.uint8)
            initial=scale_transform(np.asarray(r["transform"],np.float32),ref.shape,a.shape)
            warp,metrics=register(a,b,config.roi,initial=initial,max_width=960)
            full=scale_transform(warp,a.shape,(h,w))
            aligned=cv2.warpAffine(image,full,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT_101)
            atomic_image_write(path,aligned,[cv2.IMWRITE_TIFF_COMPRESSION,8])
            meta={"transform":full.tolist(),"alignment":metrics}
        meta.update(source_sha256=source_identity['sha256'],reference_sha256=reference_identity['sha256'])
        save_json(info,meta)
        valid=cv2.warpAffine(np.ones((h,w),np.uint8),full,(w,h),flags=cv2.INTER_NEAREST)
        valid=cv2.erode(valid,np.ones((11,11),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        common &= valid
        r.update(aligned=str(path),full_transform=full.tolist(),full_alignment=meta["alignment"])
        outputs.append(path)
        emit(stage="align",completed=i+1,total=len(records),message="Aligning selected full-resolution frames")
    return outputs,common


def fuse(paths,reference,common,config,output,emit):
    first=cv2.imread(str(paths[0]),cv2.IMREAD_UNCHANGED)
    h,w=first.shape[:2]; size=(min(1920,w),round(h*min(1920,w)/w))
    best=np.zeros(size[::-1],np.float32); second=best.copy()
    winner=np.zeros(size[::-1],np.uint16)
    score_paths=[]
    for i,path in enumerate(paths):
        score_path=path.parent/f"focus-{path.name}.npy"
        identity=image_record(path,(h,w,3))
        score_meta=load_json(score_path.with_suffix('.json'))
        score=None
        try:
            if identity and score_meta and score_meta.get('source_sha256')==identity['sha256']:
                candidate=np.load(score_path,allow_pickle=False)
                if candidate.shape==size[::-1] and candidate.dtype==np.float32 and np.isfinite(candidate).all() and (candidate>=0).all():score=candidate
        except (OSError,ValueError,EOFError):pass
        if score is None:
            image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            small=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
            score=focus_map(small)
            atomic_save_array(score_path,score)
            save_json(score_path.with_suffix('.json'),{'source_sha256':identity['sha256'] if identity else None})
        mask=score>best
        second=np.where(mask,best,np.maximum(second,score))
        winner[mask]=i
        best=np.maximum(best,score)
        score_paths.append(score_path)
    # In textureless areas choose the reference; normalized noise must not win by chance.
    informative=best>max(float(np.quantile(best,.3))*.5,2e-7)
    subject=np.ones(best.shape,np.float32)
    from .compositing import controls_active, foreground_matte
    if controls_active(config):
        # Keep focus fusion beyond the final feather so the shared compositor
        # applies opacity only once, without baking reference color into edges.
        support=(foreground_matte((h,w),config)>0).astype(np.uint8)
        support=cv2.dilate(support,np.ones((17,17),np.uint8))
        subject=cv2.resize(support.astype(np.float32),(size[0],size[1]),interpolation=cv2.INTER_AREA)
        winner[subject<.5]=reference
    elif config.subject_only:
        if config.subject_polygon:
            subject=np.zeros(best.shape,np.float32)
            points=np.rint(np.asarray(config.subject_polygon)*np.array(size)).astype(np.int32)
            cv2.fillPoly(subject,[points],1.)
            subject=cv2.GaussianBlur(subject,(0,0),max(1.,size[0]/400.))
        else:
            subject=roi_mask(best.shape,config.roi).astype(np.float32)/255.
            subject=cv2.GaussianBlur(subject,(0,0),max(2.,size[0]/160.))
        winner[subject<.5]=reference
    sums=None; weights=None
    for i,(path,score_path) in enumerate(zip(paths,score_paths)):
        score=np.load(score_path)
        relative=score/np.maximum(best,1e-12)
        weight=np.where((score>=second*.999) & informative,relative**12,0).astype(np.float32)
        if i==reference: weight[~informative]=1
        weight=cv2.GaussianBlur(weight,(0,0),1.2)
        weight*=subject
        if i==reference: weight+=1-subject
        weight=cv2.resize(weight,(w,h),interpolation=cv2.INTER_LINEAR)*common
        image=to_linear_bt709(cv2.imread(str(path),cv2.IMREAD_UNCHANGED))
        levels=min(config.pyramid_levels,int(math.log2(min(h,w)))-2)
        bands=laplacian_pyramid(image,levels)
        if sums is None:
            sums=[np.zeros_like(b) for b in bands]
            weights=[np.zeros(b.shape[:2],np.float32) for b in bands]
        for level,band in enumerate(bands):
            if level: weight=cv2.pyrDown(weight)
            sums[level]+=band*weight[:,:,None]
            weights[level]+=weight
        emit(stage="fuse",completed=i+1,total=len(paths),message="Streaming linear-light multiscale fusion")
    pyramid=[s/np.maximum(t[:,:,None],1e-12) for s,t in zip(sums,weights)]
    result=reconstruct(pyramid)
    from .regional import canvas_crop
    x0,y0,x1,y1=canvas_crop(common,config.canvas_mode)
    if config.canvas_mode=="reference":
        reference_linear=to_linear_bt709(cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED))
        result[common==0]=reference_linear[common==0]
    merged=to_srgb16(result[y0:y1,x0:x1])
    cv2.imwrite(str(output/"stack.tiff"),merged,[cv2.IMWRITE_TIFF_COMPRESSION,8])
    cv2.imwrite(str(output/"stack.png"),merged,[cv2.IMWRITE_PNG_COMPRESSION,3])
    preview=cv2.resize(merged,(min(1600,x1-x0),round((y1-y0)*min(1600,x1-x0)/(x1-x0))),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output/"preview.jpg"),(preview/256).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,95])
    labels=cv2.resize(winner,(w,h),interpolation=cv2.INTER_NEAREST)[y0:y1,x0:x1]
    if config.canvas_mode=="reference": labels[common==0]=reference
    cv2.imwrite(str(output/"source_index.png"),labels)
    confidence=np.clip(1-second/np.maximum(best,1e-12),0,1)
    confidence[(~informative)|(subject<.5)]=0
    confidence=cv2.resize(confidence,(w,h))[y0:y1,x0:x1]
    if config.canvas_mode=="reference": confidence[common==0]=0
    cv2.imwrite(str(output/"confidence.png"),(confidence*65535).astype(np.uint16))
    # Representative source and a center subject crop make visual review straightforward.
    before=to_srgb16(to_linear_bt709(cv2.imread(str(paths[reference]),cv2.IMREAD_UNCHANGED)))[y0:y1,x0:x1]
    cx0,cy0,cx1,cy1=config.roi or (.32,.17,.61,.68)
    xx0=max(0,round(cx0*w)-x0); xx1=min(x1-x0,round(cx1*w)-x0)
    yy0=max(0,round(cy0*h)-y0); yy1=min(y1-y0,round(cy1*h)-y0)
    panel=np.concatenate([before[yy0:yy1,xx0:xx1],merged[yy0:yy1,xx0:xx1]],axis=1)
    cv2.imwrite(str(output/"comparison.jpg"),(panel/256).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,97])
    return {"crop_xyxy":[x0,y0,x1,y1],"width":x1-x0,"height":y1-y0,
        "color":"sRGB transfer / BT.709 primaries; 16-bit integer outputs from 10-bit source",
        "diagnostics":{"ambiguous_fraction":float((confidence<.05).mean()),
                       "preclip_out_of_range_fraction":float(((result<0)|(result>1)).mean())}}


def run_stack(inputs,output_dir,config=None,progress=None,*,cache_dir=None,
              cache_max_bytes=DEFAULT_CACHE_MAX_BYTES,keep_frames=False):
    validate_limit(cache_max_bytes)
    if type(keep_frames) is not bool: raise ValueError("keep_frames must be a boolean")
    config=config or Config(); config.validate()
    inputs=[Path(p).resolve(strict=True) for p in inputs]
    if not inputs or any(not p.is_file() for p in inputs): raise ValueError("Input video files required")
    output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    if keep_frames and any((output/name).is_symlink() for name in ('frames','aligned')):
        raise ValueError("Frame export directories must not be symlinks")
    if any(p==output or output in p.parents for p in inputs):
        raise ValueError("Output directory must not contain source videos")
    cache=Path(cache_dir).resolve() if cache_dir is not None else output.parent/".macrostitch"/"cache"
    if cache==output or cache in output.parents or output in cache.parents or any(cache==p or cache in p.parents for p in inputs):
        raise ValueError("Cache must be separate from output and source files")
    cache.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); events=[]
    def emit(**event):
        event["elapsed_seconds"]=round(time.monotonic()-started,2)
        events.append(event)
        if progress: progress(event)
    cv2.setNumThreads(config.threads)
    cv2.setRNGSeed(0)
    with pipeline_workspace_guard(output,cache),lock(cache/"pipeline.lock"),lock(output/".pipeline.lock"),cache_retention(cache,cache_max_bytes):
        emit(stage="probe",message="Fingerprinting sources and checking video metadata")
        sources=[probe(p) for p in inputs]
        # Duplicate copies of the same clip contribute no new focus information.
        sources=list({s["sha256"]:s for s in sources}.values())
        if len({(s["width"],s["height"]) for s in sources})!=1:
            raise ValueError("All clips must have equal dimensions")
        identifier=key([VERSION,[s["sha256"] for s in sources],asdict(config)])
        manifest=output/"report.json"
        previous=load_json(manifest)
        exports_ready=True
        if keep_frames:
            selection=load_json(output/"selection.json")
            exports_ready=bool(selection and selection.get('selected'))
            if exports_ready:
                try:
                    exports_ready=all((output/directory/f"{i:03d}_clip{r['source']}_frame{r['frame']:06d}.tiff").is_file()
                                      for i,r in enumerate(selection['selected']) for directory in ('frames','aligned'))
                except (KeyError,TypeError,ValueError):exports_ready=False
        if final_cache_valid(previous,output,identifier,config) and exports_ready:
            if not keep_frames: clear_frame_exports(output)
            emit(stage="complete",message="Verified cached result")
            if previous.get('cache_key')!=identifier:
                previous.update(cache_key=identifier,config=asdict(config))
                save_json(manifest,previous)
            return {**previous,"cache_hit":True,"cache_check_seconds":round(time.monotonic()-started,2),
                    "artifacts":{name:str(output/name) for name in previous["artifacts"]}}
        records,scores,ref,analysis_dir=analyze(sources,cache,config,emit)
        if config.selection_mode=="native":
            from .native_scan import native_candidates
            from .spatial import select_native
            records,scores,ref,active_mask,native_dir=native_candidates(sources,records,ref,cache,config,emit)
            mandatory={ref}
            for region in config.local_alignment_regions or ():
                anchor=region['anchor']
                matches=[i for i,row in enumerate(records) if row['source']==anchor['source'] and row['frame']==anchor['frame']]
                if len(matches)!=1:raise ValueError('Local alignment anchor not found in the supplied videos')
                mandatory.add(matches[0])
            for anchor in config.required_frames or ():
                matches=[i for i,row in enumerate(records) if row['source']==anchor['source'] and row['frame']==anchor['frame']]
                if len(matches)!=1:raise ValueError('Required frame not found in the supplied videos')
                mandatory.add(matches[0])
            if config.max_frames<len(mandatory):raise ValueError('Frame budget cannot hold mandatory alignment anchors')
            chosen,coverage=select_native(scores,active_mask,config.max_frames,config.coverage,required=mandatory)
            save_json(output/"native-analysis.json",json.loads((native_dir/"complete.json").read_text()))
        else:
            from .compositing import selection_mask
            proxy_active=selection_mask(scores.shape[1:],config)
            chosen,coverage=select_coverage(scores,proxy_active,config.max_frames-1,config.coverage)
        if ref not in chosen: chosen.append(ref)
        chosen.sort()
        if config.selection_mode=="proxy":
            active=proxy_active.ravel()>0
            flat=scores.reshape(len(scores),-1); peak=flat.max(axis=0)
            active &= (peak>=np.quantile(peak[active],.15)) & (peak>1e-7)
            raw=flat[chosen][:,active].max(axis=0)/peak[active]
            equivalent=np.minimum(raw/.97,1)
            coverage.update(mean_coverage=float(equivalent.mean()),p10_coverage=float(np.quantile(equivalent,.1)),
                            raw_mean_coverage=float(raw.mean()),target_reached=bool(equivalent.mean()>=config.coverage))
        selected=[dict(records[i],candidate_index=i,source_sha256=sources[records[i]['source']]['sha256']) for i in chosen]
        if config.required_frames:
            required={(row['source'],row['frame']) for row in config.required_frames}
            for row in selected:row['required_frame']=(row['source'],row['frame']) in required
        reference=chosen.index(ref)
        emit(stage="select",selected=len(selected),candidates=len(records),coverage=coverage)
        from . import fusion_cache
        fusion_key=(fusion_cache.identity(config,sources,selected,reference,[VERSION,GEOMETRY_VERSION])
                    if not keep_frames else None)
        snapshot=fusion_cache.restore(cache,fusion_key,output,config,selected,reference,
                                     (sources[0]['height'],sources[0]['width']))
        if snapshot is not None:
            paths,common,result=snapshot
            emit(stage="fusion_cache",message="Verified cached spatial fusion; applying updated mask controls")
        else:
            extract(sources,selected,cache,config,emit)
            paths,common=align_selected(selected,reference,cache,config,emit)
            if config.fusion_mode=="spatial":
                from .spatial import fuse_spatial
                result=fuse_spatial(paths,reference,common,config,output,emit,selected)
            elif config.fusion_mode=="regional":
                from .regional import fuse_regional
                result=fuse_regional(paths,reference,common,config,output,emit,selected)
            else:
                result=fuse(paths,reference,common,config,output,emit)
                from .compositing import controls_active
                if config.background_mode=="transparent" and not controls_active(config):
                    from .regional import make_transparent
                    make_transparent(output,common,config,result)
            fusion_cache.save(cache,fusion_key,output,config,selected,reference,paths,common,result)
        from .compositing import compose_output
        compose_output(output,paths,reference,common,config,result)
        clear_frame_exports(output)
        if keep_frames:
            frames_dir=output/"frames"; aligned_dir=output/"aligned"
            frames_dir.mkdir(exist_ok=True); aligned_dir.mkdir(exist_ok=True)
            for i,r in enumerate(selected):
                for directory,field in [(frames_dir,"extracted"),(aligned_dir,"aligned")]:
                    dest=directory/f"{i:03d}_clip{r['source']}_frame{r['frame']:06d}.tiff"
                    if dest.is_symlink():raise ValueError(f"Unsafe frame export alias: {dest}")
                    if dest.exists() and not os.path.samefile(dest,r[field]): dest.unlink()
                    if not dest.exists(): os.link(r[field],dest)
        save_json(output/"selection.json",{"sources":sources,"selected":selected,"reference_index":reference,"coverage":coverage})
        save_json(output/"events.json",events)
        names=artifact_names(config)
        report={"version":VERSION,"cache_key":identifier,"cache_hit":False,"config":asdict(config),
            "software":{"opencv":cv2.__version__,"numpy":np.__version__,
                "ffmpeg":subprocess.check_output(["ffmpeg","-version"],text=True).splitlines()[0]},
            "sources":sources,"source_frames":sum(s["frames"] for s in sources),"candidates":len(records),
            "selected_frames":len(selected),"rejected_registration":sum(not r["accepted"] for r in records),
            "coverage":coverage,"result":result,"elapsed_seconds":round(time.monotonic()-started,2),
            "artifacts":{name:str(output/name) for name in names+["report.json"]},
            "artifact_sha256":{name:digest(output/name) for name in names},
            "limitations":[("Coverage is measured over sampled proxies, not every source frame." if config.selection_mode=="proxy" else
                             "Every source frame is examined at native subject resolution; coverage refers to structurally reliable cells, not textureless areas."),
                "Global similarity alignment cannot remove depth-dependent parallax or defocus occlusion halos.",
                ("Source-index map names the strongest source; pyramid output may blend the two strongest contributors." if config.fusion_mode=="pyramid" else
                 "Regional fusion selects real recorded surfaces; narrow seams blend neighbors. Matte and region recipe are specific to this framing." if config.fusion_mode=="regional" else
                 "Spatial labels choose recorded detail in two dimensions; multiscale seams blend contributors and low-texture areas inherit nearby source choices."),
                "No super-resolution or invented detail; focus outside the recorded rack cannot be recovered."]}
        save_json(manifest,report)
        emit(stage="complete",message="Focus stack saved",output=str(output))
        return report
