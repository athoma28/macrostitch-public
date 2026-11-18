"""Warm-only native benchmark against an existing verified reference and cache.

Generated images are discarded after comparisons; JSON/log evidence is retained.
Use --keep-artifacts to retain images for manual inspection.
No cold-cache repeatability claim is made. After a run, the existing cache is
trimmed to the default pipeline budget unless --keep-artifacts is enabled.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from macrostitch import benchmark as b
from macrostitch.pipeline import Config,digest,lock
from macrostitch.storage import DEFAULT_CACHE_MAX_BYTES,prune_cache


def evidence(output):
    output=Path(output)
    return {'output':str(output),'pixels':b.pixel_fingerprint(output/'stack.tiff'),
            'selection':b.selection_fingerprint(output/'selection.json'),
            'local_alignment_sha256':digest(output/'local-alignment.json') if (output/'local-alignment.json').exists() else None,
            'maps':{name:b.pixel_fingerprint(output/name) for name in ('source_index.png','confidence.png')}}


def compare(left,right):
    return {**b.pixel_difference(Path(left['output'])/'stack.tiff',Path(right['output'])/'stack.tiff'),
            'pixel_hash_equal':left['pixels']==right['pixels'],
            'selection_and_transforms_equal':left['selection']==right['selection'],
            'local_alignment_equal':left['local_alignment_sha256']==right['local_alignment_sha256'],
            'maps_equal':left['maps']==right['maps'],
            'map_hashes_equal':{name:left['maps'][name]==right['maps'][name] for name in left['maps']}}


def run(config_path,reference_output,output,cache,warm_runs=2,cache_runs=1,*,keep_artifacts=False):
    if warm_runs<1 or cache_runs<0:
        raise ValueError('At least one warm recomputation and nonnegative cache run count required')
    cache=Path(cache).resolve(strict=True)
    if not cache.is_dir() or not any(cache.iterdir()):
        raise ValueError('Existing populated cache directory required; no cold runs supported')
    reference_output=Path(reference_output).resolve(strict=True)
    reference_report=json.loads((reference_output/'report.json').read_text())
    config=Config(**json.loads(Path(config_path).read_text()));config.validate()
    prior=Config(**reference_report['config']);prior.validate()
    if asdict(prior)!=asdict(config):
        raise ValueError('Reference output config does not match benchmark config')
    for name,checksum in reference_report.get('artifact_sha256',{}).items():
        if digest(reference_output/name)!=checksum:
            raise ValueError(f'Reference artifact checksum mismatch: {name}')
    if not reference_report.get('artifact_sha256'):
        raise ValueError('Reference output lacks artifact checksums')
    inputs=[str(Path(s['path']).resolve(strict=True)) for s in reference_report['sources']]
    source_hashes={p:digest(p) for p in inputs}
    if any(source_hashes[str(Path(s['path']).resolve())]!=s['sha256'] for s in reference_report['sources']):
        raise ValueError('Reference source checksum mismatch')
    destination=Path(output).resolve();destination.mkdir(parents=True,exist_ok=True)
    code=b.code_fingerprints();script_sha=digest(__file__)
    reference=evidence(reference_output)
    environment_info=b.environment()
    artifacts=b.ArtifactSession(destination,keep_artifacts,protected_paths=[cache,reference_output,*inputs])
    session=artifacts.path
    report={'status':'running','session':str(session),'config':asdict(config),
            'config_sha256':b.json_hash(asdict(config)),'source_sha256':source_hashes,
            'code_sha256':code,'code_bundle_sha256':b.json_hash(code),'benchmark_script_sha256':script_sha,
            'cache_limit_bytes':None if keep_artifacts else DEFAULT_CACHE_MAX_BYTES,
            'keep_artifacts':keep_artifacts,'environment':environment_info,'existing_cache':str(cache),'reference':reference,'runs':[],
            'limitations':['Warm application-cache recomputations only; cold repeatability is untested.',
                          'OS caches uncontrolled. Existing intermediate cache may be completed by the pipeline.',
                          'Pipeline timing excludes post-run evidence hashing; memory is worker/child peak RSS separately.']}
    b.write_json(session/'benchmark.json',report)
    try:
        for index in range(warm_runs+cache_runs):
            mode='warm_recompute' if index<warm_runs else 'final_cache'
            run_output=session/f'warm-{index:03d}' if index<warm_runs else session/f'warm-{warm_runs-1:03d}'
            artifacts.register(run_output)
            request={'inputs':inputs,'config':asdict(config),'mode':mode,'cache':str(cache),'output':str(run_output)}
            row=b.launch(request,session/f'run-{index:03d}-{mode}')
            row.update(evidence(run_output))
            row['comparison_to_reference']=compare(reference,row)
            report['runs'].append(row)
            b.write_json(session/'benchmark.json',report)
        report['repeatability']=b.compare_runs(report['runs'])
        report['timing_summary']=b.timing_summary(report['runs'])
        report['all_runs_match_reference']=all(all(r['comparison_to_reference'][key] for key in
            ('pixels_equal','pixel_hash_equal','selection_and_transforms_equal','maps_equal','local_alignment_equal')) for r in report['runs'])
        report['provenance_unchanged']=(code==b.code_fingerprints() and script_sha==digest(__file__) and
            source_hashes=={p:digest(p) for p in inputs} and reference==evidence(reference_output))
        if not report['provenance_unchanged']:
            raise RuntimeError('Source, code, or reference changed during benchmark')
        report['status']='complete'
    except Exception:
        report.update(status='error',error=traceback.format_exc())
        raise
    finally:
        try:
            artifacts.close()
            if not keep_artifacts:
                with lock(cache/'pipeline.lock'):
                    report['cache_retention']=prune_cache(cache,DEFAULT_CACHE_MAX_BYTES)
        finally:
            b.write_json(session/'benchmark.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('config','reference-output','output','cache'):
        parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--warm-runs',type=int,default=2)
    parser.add_argument('--cache-runs',type=int,default=1)
    parser.add_argument('--keep-artifacts',action='store_true',help='Retain generated images for manual inspection')
    args=parser.parse_args()
    report=run(args.config,args.reference_output,args.output,args.cache,args.warm_runs,args.cache_runs,keep_artifacts=args.keep_artifacts)
    print(json.dumps({'report':str(Path(report['session'])/'benchmark.json'),
                      'all_runs_match_reference':report['all_runs_match_reference']},indent=2))


if __name__=='__main__':
    main()
