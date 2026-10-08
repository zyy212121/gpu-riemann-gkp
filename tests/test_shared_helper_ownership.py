"""Keep extracted shared algorithms behind one implementation owner."""
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('helper,consumer', [
 ('GpuAutomaticCsrScheduleFields.inl','applications/gasUGKP/private_backend/GpuResidentStrict.cu'),
 ('GpuAutomaticCsrSchedule.cuh','applications/gasUGKP/private_backend/GpuResidentStrict.cu'),
 ('GpuBlockComponentReduction.cuh','applications/gasUGKP/private_backend/GpuResidentStrict.cu'),
 ('GpuParticleDirectoryHostPolicy.cuh','common/GpuParticleDirectoryHost.cuh'),
 ('GpuPressureUnsortedAlgebra.cuh','common/GpuPressureKickAccumulation.cuh'),
])
def test_shared_helper_has_real_owner_and_consumer(helper,consumer):
    assert (ROOT/'common'/helper).is_file(),helper
    assert '#include "'+helper+'"' in (ROOT/consumer).read_text()
