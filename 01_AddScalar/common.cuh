#include <NvInfer.h>
#include <string>
#include <vector>
#include <map>


using namespace nvinfer1;

// plugin debug function
#ifdef DEBUG
    #define WHERE_AM_I()                          
        do                                        
        {                                        
            printf("%14p[%s]\n", this, __func__);
        } while (0);
#else
    #define WHERE_AM_I()
#endif

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))

// get the size in byte of a TensorRT data type
__inline__ size_t dataTypeToSize(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
        return 1;
    case DataType::kINT32:
        return 4;
    case DataType::kBOOL:
        return 1;
    case DataType::kUINT8:
        return 1;
    case DataType::kFP8:
        return 1;
    default:
        return 4;
    }
}

// get the string of a TensorRT data format
__inline__ std::string formatToString(TensorFormat format)
{
    switch (format)
    {
    case TensorFormat::kLINEAR:
        return std::string("LINE ");
    case TensorFormat::kCHW2:
        return std::string("CHW2 ");
    case TensorFormat::kHWC8:
        return std::string("HWC8 ");
    case TensorFormat::kCHW4:
        return std::string("CHW4 ");
    case TensorFormat::kCHW16:
        return std::string("CHW16");
    case TensorFormat::kCHW32:
        return std::string("CHW32");
    case TensorFormat::kHWC:
        return std::string("HWC  ");
    case TensorFormat::kDLA_LINEAR:
        return std::string("DLINE");
    case TensorFormat::kDLA_HWC4:
        return std::string("DHWC4");
    case TensorFormat::kHWC16:
        return std::string("HWC16");
    default: return std::string("None ");
    }
}

