#include "common.cuh"


// 匿名命名空间，该命名空间中的内容仅在当前文件中可见
namespace
{
static const char *PLUGIN_NAME {"AddScalar"};
static const char *PLUGIN_VERSION {"1"};
}


namespace nvinfer1
{
class AddScalarPlugin : public IPluginV2DynamicExt
{
private:
    const std::string name_;
    std::string       namespace_;
    struct
    {
        float scalar;
    } m_;

public:
    // 构造函数接受 PluginCreator 的参数，做一些构建期的初始化工作
    AddScalarPlugin(const std::string &name, float scalar);
    AddScalarPlugin(const std::string &name, const void *buffer, size_t length);
    ~AddScalarPlugin();

    // Method inherited from IPluginV2
    const char *getPluginType() const noexcept override;
    const char *getPluginVersion() const noexcept override;
    int32_t     getNbOutputs() const noexcept override;
    int32_t     initialize() noexcept override;
    void        terminate() noexcept override;
    size_t      getSerializationSize() const noexcept override;
    void        serialize(void *buffer) const noexcept override;
    void        destroy() noexcept override;
    void        setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char *getPluginNamespace() const noexcept override;

    // Method inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept override;
    void     attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept override;
    void     detachFromContext() noexcept override;

    // Method inherited from IPluginV2DynamicExt
    IPluginV2DynamicExt *clone() const noexcept override;
    DimsExprs            getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override;
    bool                 supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void                 configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override;
    size_t               getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override;
    int32_t              enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

protected:
    // To prevent compiler warnings
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2Ext::configurePlugin;
};

class AddScalarPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    AddScalarPluginCreator();
    ~AddScalarPluginCreator();
    const char                  *getPluginName() const noexcept override;
    const char                  *getPluginVersion() const noexcept override;
    const PluginFieldCollection *getFieldNames() noexcept override;
    IPluginV2DynamicExt         *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override;
    IPluginV2DynamicExt         *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;
    void                         setPluginNamespace(const char *pluginNamespace) noexcept override;
    const char                  *getPluginNamespace() const noexcept override;
};

}