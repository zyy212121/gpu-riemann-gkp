"""Execute production automatic scheduling control flow with host CUDA API stubs.

This verifies decision/task-publication logic, not GPU reductions or execution.
"""
from pathlib import Path
import re
import subprocess
ROOT = Path(__file__).resolve().parents[1]


def test_auto_decision_guarantees_ready_tasks_and_preserves_metadata(tmp_path):
    common = ROOT/'common/GpuAutomaticCsrSchedule.cuh'
    name = 'runAutomaticCsrSchedule' if common.exists() else 'runToolB3'
    text = common.read_text() if common.exists() else (ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    start = text.index('int '+name+'\n' if name=='runToolB3' else 'int '+name+'\n')
    opening = text.index('{',start);end=opening+1;depth=1
    while depth:
        depth += (text[end]=='{')-(text[end]=='}');end+=1
    body = re.sub(r'<<<[^>]+>>>','',text[start:end])
    source = r'''
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
enum class HeavyDirectoryKind {full=0,splitBaseAndInjection=1,baseOnly=2};
#define GPU_DIRECTORY_PARAMETER_TYPE HeavyDirectoryKind
#define GPU_DIRECTORY_SELECTOR directoryKind
#define GPU_DIRECTORY_ARGUMENT static_cast<int>(directoryKind)
#define GPU_AUTO_THRESHOLD_FIELD csrHeavyCellThreshold
#define GPU_AUTO_UPDATE_POLICY(s,k) configureDynamicCsrHeavyPolicy(s,k)
struct DeviceState {
 int csrHeavyReductionMode=2,particleCapacity=100,csrHeavyAutoInterval=3,nCells=4;
 unsigned long long schedulingAdvanceCount=0;
 int csrHeavyCellThreshold=32,csrHeavyTileParticles=32,csrHeavyReductionActive=0,csrHeavyReductionEnabled=0;
 int csrTasksReady=0,csrPreparedDirectoryKind=-1,maximum=0,tasks=0,cells=0;
 int *csrMaximumOccupancy=&maximum,*csrHeavyTaskCount=&tasks,*csrHeavyCellCount=&cells;
 DeviceState* deviceState=this;
};
using cudaError_t=int;const int cudaSuccess=0,cudaErrorInvalidValue=1,cudaMemcpyDeviceToHost=0;
int occupancy=64,prepared=0,policyCalls=0;
int cudaMemset(void*p,int v,size_t n){std::memset(p,v,n);return 0;}
int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
int cudaGetLastError(){return 0;}
void setLastError(const char*,int){}
int configureDynamicCsrHeavyPolicy(DeviceState*,HeavyDirectoryKind){++policyCalls;return 0;}
void maximumDirectoryOccupancyKernel(DeviceState*,int,int* out){*out=occupancy;}
void publishHeavyReductionDecisionKernel(DeviceState*s,int active){s->csrHeavyReductionActive=s->csrHeavyReductionEnabled=active;}
int prepareCsrSegmentedReductionTasks(DeviceState*s,int,HeavyDirectoryKind kind){
 ++prepared;s->csrTasksReady=1;s->csrPreparedDirectoryKind=int(kind);s->tasks=3;s->cells=1;return 0;
}
@BODY@
int main(){
 DeviceState s;
 assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0);
 if(prepared!=1 || s.tasks!=3 || s.cells!=1 || s.csrTasksReady!=1){std::cerr<<"active decision did not publish executable task partition";return 1;}
 // A new producer invalidates readiness even on a non-inspection step.
 s.csrTasksReady=0;
 assert(@NAME@(&s,128,HeavyDirectoryKind::baseOnly)==0);
 assert(prepared==2 && s.csrPreparedDirectoryKind==2 && policyCalls==1);
 // Low occupancy disables L2 and clears executable metadata, not occupancy aliases.
 occupancy=8;assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0);
 assert(!s.csrHeavyReductionEnabled && s.tasks==0 && s.cells==0);
 // No inspection on steps 4/5; step 6 recovers ready tasks when occupancy grows.
 occupancy=80;assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0);
 assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0);
 assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0);
 assert(s.csrHeavyReductionEnabled && prepared==3 && s.tasks==3 && s.cells==1);
 // Explicit L2 is untouched by automatic scheduling.
 s.csrHeavyReductionMode=1;auto count=s.schedulingAdvanceCount;
 assert(@NAME@(&s,128,HeavyDirectoryKind::full)==0 && s.schedulingAdvanceCount==count);
}
'''.replace('@BODY@',body).replace('@NAME@',name)
    cpp=tmp_path/'schedule.cpp';cpp.write_text(source);exe=tmp_path/'schedule'
    q=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
