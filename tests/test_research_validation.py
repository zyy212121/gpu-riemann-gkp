import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
    spec=importlib.util.spec_from_file_location('research',ROOT/'tools/run_research_matrix.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def row():
    return dict(status='ok',timing_valid='1',base_particle_count='12',injected_particle_count='3',pretransport_particle_count='15',removed_particle_count='2',particle_count='13',occupancy_sum='13',occupancy_matches_count='1',bad_cells='0',bad_particles='0',bad_field_mask='0',total_ms='1.5',particle_path='1',heavy_reduction_enabled='1')

def test_valid_population_ledger_and_timing_accepted():
    assert module().validate_rows([row()],1,1)['total_ms']==1.5

@pytest.mark.parametrize('key,value',[('occupancy_sum','12'),('removed_particle_count','0'),('bad_particles','1'),('total_ms','nan'),('timing_valid','0'),('heavy_reduction_enabled','0')])
def test_invalid_run_is_not_reported_as_performance(key,value):
    r=row();r[key]=value
    with pytest.raises(ValueError):module().validate_rows([r],1,1)

def test_absent_cuda_probe_is_rejected():
    with pytest.raises(ValueError):module().validate_rows([],1,1)
