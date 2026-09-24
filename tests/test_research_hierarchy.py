from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def test_research_states_keep_each_paper_layer_independent(tmp_path):
    source=tmp_path/'test.cpp'
    source.write_text(r'''#include "GpuResearchHierarchy.H"
#include <cassert>
int main() {
  struct Row { const char* name; bool cell,warp,split,heavy; };
  const Row rows[]={{"L0",0,0,0,0},{"L1",1,0,0,0},{"T1",1,1,0,0},{"E1",1,1,0,0},{"S1",1,1,1,0},{"T2",1,1,0,1},{"E2",1,1,0,1},{"S2",1,1,1,1}};
  for(const auto& r:rows) { gpuResearch::Flags f{}; assert(gpuResearch::decode(r.name,f)); assert(f.cell==r.cell); assert(f.warp==r.warp); assert(f.split==r.split); assert(f.heavy==r.heavy); }
  gpuResearch::Flags f{true,false,true,false}; assert(!gpuResearch::decode("unknown",f)); assert(f.cell && !f.warp && f.split && !f.heavy);
}
''')
    exe=tmp_path/'test'
    subprocess.run(['g++','-std=c++14','-I',str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
