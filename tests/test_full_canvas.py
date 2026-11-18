"""Small native-resolution fixtures verify canvas extent and valid fusion bounds."""
import cv2
import numpy as np
import pytest

from macrostitch.pipeline import Config, fuse
from macrostitch.regional import canvas_crop, fuse_regional
from macrostitch.imaging import to_linear_bt709, to_srgb16, common_crop


@pytest.fixture
def scene(tmp_path):
    h,w=64,96
    yy,xx=np.indices((h,w))
    reference=np.repeat((9000+xx*140+yy*70)[...,None],3,axis=2).astype(np.uint16)
    sharp=np.repeat(np.where((xx//2+yy//2)%2,44000,12000)[...,None],3,axis=2).astype(np.uint16)
    paths=[]
    for i,image in enumerate((reference,sharp)):
        directory=tmp_path/f'frame-{i}'; directory.mkdir()
        path=directory/'aligned.tiff'
        assert cv2.imwrite(str(path),image)
        paths.append(path)
    common=np.zeros((h,w),np.uint8)
    common[8:-8,12:-12]=1
    # An interior invalid hole must also use reference, not a reflected aligned sample.
    common[25:28,25:28]=0
    return paths,common,reference


@pytest.mark.parametrize('method',['pyramid','regional'])
@pytest.mark.parametrize('mode',['common','reference'])
def test_full_canvas_extent_and_reference_background(scene,tmp_path,method,mode):
    paths,common,reference=scene
    config=Config(canvas_mode=mode,background_mode='reference',subject_only=False,
                  pyramid_levels=2,roi=(.2,.2,.8,.8),fusion_mode=method,
                  focus_regions=({'y':[0,1],'roi':[.3,.3,.7,.7]},) if method=='regional' else None)
    config.validate()
    output=tmp_path/'output'; output.mkdir()
    if method=='regional':
        records=[{'source':0,'frame':i,'timestamp':i/10} for i in range(2)]
        result=fuse_regional(paths,0,common,config,output,lambda **event:None,records)
    else:
        result=fuse(paths,0,common,config,output,lambda **event:None)
    expected=(0,0,96,64) if mode=='reference' else common_crop(common)
    assert result['crop_xyxy']==list(expected)
    x0,y0,x1,y1=expected
    merged=cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
    assert merged.shape==(y1-y0,x1-x0,3)
    if mode=='reference':
        expected_reference=to_srgb16(to_linear_bt709(reference))
        np.testing.assert_array_equal(merged[common==0],expected_reference[common==0])
        assert np.any(merged[common>0]!=expected_reference[common>0])
        confidence=cv2.imread(str(output/'confidence.png'),cv2.IMREAD_UNCHANGED)
        assert not confidence[common==0].any()


def test_canvas_helper_and_validation():
    assert Config().canvas_mode=='common'
    with pytest.raises(ValueError,match='canvas_mode'):
        Config(canvas_mode='crop').validate()
    with pytest.raises(ValueError,match='canvas_mode'):
        canvas_crop(np.ones((3,4),np.uint8),'crop')
    assert canvas_crop(np.ones((3,4),np.uint8),'reference')==(0,0,4,3)


@pytest.mark.parametrize('method',['pyramid','regional'])
def test_reference_canvas_transparency_keeps_extent_and_masks_invalid(scene,tmp_path,method):
    from macrostitch.regional import make_transparent
    paths,common,_=scene
    config=Config(canvas_mode='reference',background_mode='transparent',subject_only=False,
                  subject_polygon=((0,0),(1,0),(1,1),(0,1)),pyramid_levels=2,
                  roi=(.2,.2,.8,.8),fusion_mode=method,
                  focus_regions=({'y':[0,1],'roi':[.3,.3,.7,.7]},) if method=='regional' else None)
    config.validate()
    output=tmp_path/'transparent'; output.mkdir()
    if method=='regional':
        records=[{'source':0,'frame':i,'timestamp':i/10} for i in range(2)]
        result=fuse_regional(paths,0,common,config,output,lambda **event:None,records)
    else:
        result=fuse(paths,0,common,config,output,lambda **event:None)
        make_transparent(output,common,config,result)
    rgba=cv2.imread(str(output/'stack.png'),cv2.IMREAD_UNCHANGED)
    assert rgba.shape==(64,96,4)
    assert result['crop_xyxy']==[0,0,96,64]
    assert not rgba[...,3][common==0].any()
    assert rgba[...,3][common>0].max()==65535
