#include <NvInfer.h>
#include <fstream>
#include <iostream>

using namespace nvinfer1;


#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine << " in file " << szFile << std::endl;
        return false;
    }
    return true;
}


class Logger : public ILogger
{
public:
    Severity reportableSeverity;

    Logger(Severity severity = Severity::kINFO):
        reportableSeverity(severity) {}

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > reportableSeverity)
        {
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

__inline__ std::string shapeToString(Dims32 dim)
{
    std::string output("[");
    if (dim.nbDims == 0)
    {
        return output + std::string("]");
    }
    for (int i = 0; i < dim.nbDims - 1; ++i)
    {
        output += std::to_string(dim.d[i]) + std::string(", ");
    }
    output += std::to_string(dim.d[dim.nbDims - 1]) + std::string("]");
    return output;
}

__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}