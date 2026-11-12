"""Regression checks against the exported mesh, not just builder counters."""
import numpy as np
import pytest

from phidias_hoard.engine import meshio, sculpt, preview
from phidias_hoard.engine.finish import find_blender
from phidias_hoard.remodel import build, pieces


def piece(kind, params, closed=False):
    points = [[-5,-4,0],[5,-4,0],[5,4,0],[-5,4,0]]
    triangles = [[0,1,2],[0,2,3]]
    if closed:
        points += [[-5,-4,3],[5,-4,3],[5,4,3],[-5,4,3]]
        triangles += [[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],
                      [2,6,7],[2,7,3],[3,7,4],[3,4,0]]
    return dict(id=1, name="surface", kind=kind, params=params, center=[0,0,0],
                size=[10,8,3 if closed else 0], points=points, triangles=triangles)


@pytest.fixture(scope="module")
def blender():
    exe=find_blender()
    if not (exe or build.has_bpy()):
        pytest.skip("Blender is not installed")
    return exe


def test_requested_sheet_thickness_survives_obj_export(tmp_path, blender):
    for kind in ("keep", "organic", "rock"):
        out=tmp_path/kind
        stats=build.run({"pieces":[piece(kind,dict(level=1, thickness_mm=0.8, bevel=0))]},out,blender)
        P,F,_=preview.read_polygons(out/"remodel.obj")
        assert sculpt.open_edges(F,len(P))==0
        assert np.ptp(P[:,2])==pytest.approx(0.8,abs=1e-5)
        assert stats["pieces"][0]["thickness_mm"]==0.8
        assert stats["pieces"][0]["boundary_edges_after"]==0
        assert not stats["pieces"][0]["ngons"]
        assert not any("piece failed" in line for line in stats["log"])


def test_closed_piece_is_not_given_an_inner_shell(tmp_path, blender):
    for thickness in (0,0.8):
        build.run({"pieces":[piece("keep",dict(thickness_mm=thickness, bevel=0),closed=True)]},tmp_path/str(thickness),blender)
    a=meshio.load_model(tmp_path/"0/remodel.obj")
    b=meshio.load_model(tmp_path/"0.8/remodel.obj")
    np.testing.assert_array_equal(a.P,b.P)
    np.testing.assert_array_equal(a.F,b.F)


def test_rock_quads_preserve_ridge_and_open_outline(tmp_path, blender):
    q=piece("rock",dict(level=2,emboss=0))
    q["points"]=[[-5,-4,0],[0,-4,3],[5,-4,0],[-5,4,0],[0,4,3],[5,4,0]]
    q["triangles"]=[[0,1,4],[0,4,3],[1,2,5],[1,5,4]]
    q["size"]=[10,8,3]
    stats=build.run({"pieces":[q]},tmp_path,blender)
    P,F,_=preview.read_polygons(tmp_path/"remodel.obj")
    # Every exported vertex lies on one of the two original planes.
    assert np.max(np.abs(P[:,2]-(3-0.6*np.abs(P[:,0]))))<1e-5
    np.testing.assert_allclose(np.ptp(P,axis=0),q["size"],atol=1e-5)
    assert sculpt.open_edges(F,len(P))>0
    assert stats["pieces"][0]["tris"]==stats["pieces"][0]["ngons"]==0
    assert stats["pieces"][0]["quads"]>4


def test_volume_parameters_are_available_and_clamped():
    assert "rock" in pieces.allowed(dict(triangles_count=2,fits=dict(rect=False,round=False)))
    assert pieces.clean_params("rock",dict(level=99,thickness_mm=99,emboss=0))==dict(level=4,emboss=0,thickness_mm=5)
    assert pieces.clean_params("organic",dict(thickness_mm=-2))==dict(thickness_mm=0)


def test_folded_sheet_offsets_stay_bounded(tmp_path, blender):
    q=piece("keep",dict(thickness_mm=0.8,bevel=0))
    q["points"]=[[-5,-4,0],[0,-4,0],[0,4,0],[-5,4,0],[-4.999,-4,0.001],[-4.999,4,0.001]]
    q["triangles"]=[[0,1,2],[0,2,3],[1,4,5],[1,5,2]]
    q["size"]=[5,8,0.001]
    build.run({"pieces":[q]},tmp_path,blender)
    P,_,_=preview.read_polygons(tmp_path/"remodel.obj")
    source=np.array(q["points"])
    assert np.all(P.min(0)>=source.min(0)-0.40001)
    assert np.all(P.max(0)<=source.max(0)+0.40001)


def test_named_rock_does_not_use_figure_rounding():
    q=dict(dropped=False,profile=[[0,2,2],[3,1,2]],size=[4,4,3],figure=True,
           fits=dict(rect=False,round=False),materials={"gura_iwa.png":{}})
    assert pieces.guess(q,150)[0]=="rock"
    q["materials"]={"roof_houou03_hou":{}}
    assert pieces.guess(q,150)[0]=="organic"


def test_branching_game_surface_gets_separate_rims(tmp_path, blender):
    q=piece("rock",dict(level=1,thickness_mm=0.2,emboss=0))
    q["points"]=[[0,-2,0],[0,2,0],[-3,0,0],[3,0,0],[0,0,3]]
    q["triangles"]=[[0,1,2],[1,0,3],[0,1,4]]
    q["size"]=[6,4,3]
    stats=build.run({"pieces":[q]},tmp_path,blender)
    P,F,_=preview.read_polygons(tmp_path/"remodel.obj")
    assert sculpt.open_edges(F,len(P))==0
    edges=np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    assert np.all(counts==2)
    assert stats["pieces"][0]["split_branch_edges"]>0
    assert stats["pieces"][0]["thickness_mm"]==0.2
