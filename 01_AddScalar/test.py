import ctypes
import numpy as np
import tensorrt as trt
from cuda import cudart


scalar = 1
ctypes.cdll.LoadLibrary('lib/libAddScalarPlugin.so')

logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
profile = builder.create_optimization_profile()
config = builder.create_builder_config()

input_tensor = network.add_input('input', trt.float32, (-1, 1, 3, 3))
profile.set_shape(input_tensor.name, [1, 1, 3, 3], [4, 1, 3, 3], [8, 1, 3, 3])
config.add_optimization_profile(profile)

add_scalar_param = trt.PluginField('scalar', np.float32(scalar), trt.PluginFieldType.FLOAT32)
add_scalar_creator = trt.get_plugin_registry().get_plugin_creator('AddScalar', '1')
add_scalar_plugin = add_scalar_creator.create_plugin(add_scalar_creator.name, trt.PluginFieldCollection([add_scalar_param]))
add_scalar_layer = network.add_plugin_v2([input_tensor], add_scalar_plugin)
network.mark_output(add_scalar_layer.get_output(0))

engineString = builder.build_serialized_network(network, config)
if engineString is None:
    print("Failed building engine!")
    exit()
print("Succeeded building engine!")

engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

context = engine.create_execution_context()
context.set_input_shape(input_name, [1, 1, 3, 3])
input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)
input_dtype = engine.get_tensor_dtype(input_name)
output_dtype = engine.get_tensor_dtype(output_name)

inputHost = np.arange(np.prod(input_shape), dtype=trt.nptype(input_dtype)).reshape(input_shape)
outputHost = np.empty(output_shape, dtype=trt.nptype(output_dtype))
_, inputDevice = cudart.cudaMalloc(inputHost.nbytes)
_, outputDevice = cudart.cudaMalloc(outputHost.nbytes)
context.set_tensor_address(input_name, inputDevice)
context.set_tensor_address(output_name, outputDevice)

cudart.cudaMemcpy(inputDevice, inputHost.ctypes.data, inputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
context.execute_async_v3(0)
cudart.cudaMemcpy(outputHost.ctypes.data, outputDevice, outputHost.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

cudart.cudaFree(inputDevice)
cudart.cudaFree(outputDevice)

print(f'Input: {inputHost.flatten()}')
print(f'Output: {outputHost.flatten()}')
