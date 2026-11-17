#include "bout/bout.hxx"
#include "bout/build_config.hxx"
#include "bout/difops.hxx"
#include "bout/field.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/fv_ops.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/utils.hxx"

#include <fmt/format.h>

#include <cmath>
#include <string>

namespace {
auto fci_op_test(const std::string& name, Options& dump, const Field3D& input,
                 const Field3D& result) {
  auto* mesh = input.getMesh();
  const Field3D solution{FieldFactory::get()->create3D(fmt::format("{}_solution", name),
                                                       Options::getRoot(), mesh)};
  const Field3D error{result - solution};

  dump[fmt::format("{}_l_2", name)] = sqrt(mean(SQ(error), true, "RGN_NOBNDRY"));
  dump[fmt::format("{}_l_inf", name)] = max(abs(error), true, "RGN_NOBNDRY");

  dump[fmt::format("{}_result", name)] = result;
  dump[fmt::format("{}_error", name)] = error;
  dump[fmt::format("{}_input", name)] = input;
  dump[fmt::format("{}_solution", name)] = solution;

  for (int slice = 1; slice < mesh->ystart; ++slice) {
    dump[fmt::format("{}_input.ynext(-{})", name, slice)] = input.ynext(-slice);
    dump[fmt::format("{}_input.ynext({})", name, slice)] = input.ynext(slice);
  }
}
} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  using bout::globals::mesh;

  Field3D input{FieldFactory::get()->create3D("input_field", Options::getRoot(), mesh)};
  Field3D K{FieldFactory::get()->create3D("K", Options::getRoot(), mesh)};

  // Communicate to calculate parallel transform.
  mesh->communicate(input, K);

  Options dump;
  // Add mesh geometry variables
  mesh->outputVars(dump);

  // Dummy variable for *_mod overloads
  Field3D flow_ylow;

  fci_op_test("grad_par", dump, input, Grad_par(input));
  fci_op_test("grad2_par2", dump, input, Grad2_par2(input));
  fci_op_test("div_par", dump, input, Div_par(input));
  fci_op_test("div_par_K_grad_par", dump, input, Div_par_K_Grad_par(K, input));
  fci_op_test("div_par_K_grad_par_mod", dump, input,
              Div_par_K_Grad_par_mod(K, input, flow_ylow));
  fci_op_test("laplace_par", dump, input, Laplace_par(input));

  // Finite volume methods
  fci_op_test("FV_div_par_mod", dump, input, FV::Div_par_mod(input, K, K, flow_ylow));
  fci_op_test("FV_div_par_fvv", dump, input, FV::Div_par_fvv(input, K, K));

  // Perpendicular diffusion (poloidal-plane form on unstructured meshes)
  const auto dagp = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
  const Field3D dagp_result = (*dagp)(K, input, false);
  fci_op_test("FV_div_a_grad_perp", dump, input, dagp_result);

  // Conservation: every plane is closed (periodic in Z, zero-flux walls in
  // R), so the volume integral vanishes
  Field3D volume{mesh};
  mesh->get(volume, "dagp_us_volume", 0.0, false);
  dump["FV_div_a_grad_perp_integral"] =
      mean(volume * dagp_result, true, "RGN_NOBNDRY")
      / mean(volume * abs(dagp_result), true, "RGN_NOBNDRY");

  bout::writeDefaultOutputFile(dump);

  BoutFinalise();
}
