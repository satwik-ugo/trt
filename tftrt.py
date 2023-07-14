from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Conversion Parameters 
conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP16)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir="./models/saved_model/",
    conversion_params=conversion_params)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()


# Save the model to the disk 
converter.save("./models/optmized_models")