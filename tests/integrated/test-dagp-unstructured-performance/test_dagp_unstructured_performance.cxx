// Cost of the unstructured perpendicular operator Div_a_Grad_perp_FCI
// (time per call). See test_dagp_unstructured_performance.py.

#include "bout/bout.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/fv_ops.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"

#include <chrono>

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  {
    using bout::globals::mesh;
    Field3D f = FieldFactory::get()->create3D("f", Options::getRoot(), mesh);
    Field3D a = FieldFactory::get()->create3D("a", Options::getRoot(), mesh);
    mesh->communicate(f, a);

    const int ncalls = Options::root()["ncalls"].withDefault(200);
    const auto dagp = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
    Field3D flow_R;
    Field3D flow_Z;
    Field3D result = (*dagp)(a, f, flow_R, flow_Z, false); // warm-up

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    for (int i = 0; i < ncalls; ++i) {
      result = (*dagp)(a, f, flow_R, flow_Z, false);
    }
    const double elapsed = std::chrono::duration<double>(clock::now() - start).count();

    Options dump;
    dump["time_per_call"] = elapsed / ncalls;
    dump["result"] = result;
    bout::writeDefaultOutputFile(dump);
  }

  BoutFinalise();
  return 0;
}
