#include <torch/script.h>
#include <trtorch/trtorch.h>
using namespace std;

void compile() {
  constexpr int64_t INPUT_CHANNEL_NUM = 256;
  constexpr int64_t WIDTH = 32;
  torch::jit::Module module = torch::jit::load("model.ts");
  module.to(torch::kCUDA);
  module.eval();

  std::vector<int64_t> in_sizes = {1, INPUT_CHANNEL_NUM, WIDTH, WIDTH};
  trtorch::CompileSpec::InputRange range(in_sizes);
  trtorch::CompileSpec info({range});
  module = trtorch::CompileGraph(module, info);
}

int main() {
  for (int64_t i = 0; i < 10000; i++) {
    cout << i << endl;
    compile();
  }
}