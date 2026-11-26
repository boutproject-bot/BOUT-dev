// Div(a Grad_perp f) on an unstructured mesh, with f and a scaled on each z
// plane k by (1 + k) and (1 + k / 2). See test_unstructured_planes.py.

#include "bout/bout.hxx"
#include "bout/boutexception.hxx"
#include "bout/constants.hxx"
#include "bout/field3d.hxx"
#include "bout/fv_ops.hxx"
#include "bout/globals.hxx"
#include "bout/mesh.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"

#include <cmath>

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  {
    Mesh* mesh = bout::globals::mesh;
    Field3D R{mesh};
    Field3D Z{mesh};
    if (mesh->get(R, "R", 0.0, false) != 0 or mesh->get(Z, "Z", 0.0, false) != 0) {
      throw BoutException("R and Z not in the grid file");
    }
    Field3D f{zeroFrom(R)};
    Field3D a{zeroFrom(R)};
    // Including the wall nodes in the x guard cells (boundary values)
    BOUT_FOR(i, f.getRegion("RGN_ALL")) {
      const auto k = static_cast<BoutReal>(i.z());
      f[i] = std::sin(2 * R[i]) * std::cos(2 * PI * Z[i]) * (1 + k);
      a[i] = (1 + 0.3 * R[i]) * (1 + 0.5 * k);
    }

    const auto dagp = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
    Field3D flow_R;
    Field3D flow_Z;
    Field3D wall;
    Options dump;
    dump["result"] = (*dagp)(a, f, flow_R, flow_Z, wall, false);
    dump["result_upwind"] = (*dagp)(a, f, true);
    dump["flow_R"] = flow_R;
    dump["flow_Z"] = flow_Z;
    dump["wall"] = wall;
    bout::writeDefaultOutputFile(dump);
  }
  BoutFinalise();
  return 0;
}
