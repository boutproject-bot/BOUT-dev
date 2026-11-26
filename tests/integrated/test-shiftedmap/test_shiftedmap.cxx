// Parallel slices and Grad_par of a field read from the grid file, with the
// parallel transform chosen in the input (shifted or shiftedmap). Writes,
// for each cell, the values of its forward and backward slices (which are
// stored at y + 1 and y - 1) so that runs on different layouts of the same
// cells can be compared cell by cell.

#include "bout/bout.hxx"
#include "bout/difops.hxx"
#include "bout/field3d.hxx"
#include "bout/globals.hxx"
#include "bout/mesh.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);
  {
    Mesh* mesh = bout::globals::mesh;
    Field3D f{mesh};
    if (mesh->get(f, "f_test", 0.0, false) != 0) {
      throw BoutException("f_test not in the grid file");
    }
    mesh->communicate(f); // Also calculates the parallel slices

    Field3D up{zeroFrom(f)};
    Field3D down{zeroFrom(f)};
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      up[i] = f.yup()[i.yp()];
      down[i] = f.ydown()[i.ym()];
    }

    Options dump;
    dump["f_up"] = up;
    dump["f_down"] = down;
    dump["grad_par"] = Grad_par(f);
    bout::writeDefaultOutputFile(dump);
  }
  BoutFinalise();
  return 0;
}
