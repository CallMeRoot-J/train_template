import torch


def export_pytorch(output_dir, model):
    state_dicts = {'model': model.state_dict()}
    torch.save(state_dicts, output_dir)
    print(f"Pytorch model has been written to {output_dir}")


def export_jit(data, output_dir, model):
    jit_model = torch.jit.trace(model, data)
    torch.jit.save(jit_model, output_dir)
    print(f"Jit model has been written to {output_dir}")


def export_onnx(data, output_dir, model):
    class ONNXModelWrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, board_size, board_input, stm_input):
            value, policy = self.model({
                'board_size': board_size,
                'board_input': board_input,
                'stm_input': stm_input,
            })
            return value, policy

    model = ONNXModelWrapper(model)
    model.eval()
    args = (
        data['board_size'],
        data['board_input'],
        data['stm_input'],
    )
    torch.onnx.export(model,
                      args,
                      output,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['board_size', 'board_input', 'stm_input'],
                      output_names=['value', 'policy'],
                      dynamic_axes={
                          'board_size': {
                              0: 'batch_size'
                          },
                          'board_input': {
                              0: 'batch_size'
                          },
                          'stm_input': {
                              0: 'batch_size'
                          },
                          'value': {
                              0: 'batch_size'
                          },
                          'policy': {
                              0: 'batch_size'
                          },
                      })
    print(f"Onnx model has been written to {output}.")
