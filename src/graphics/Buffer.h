#ifndef GRAPHICS_BUFFER_H
#define GRAPHICS_BUFFER_H

#include <memory>
#include <string_view>

class Buffer
{
public:
    enum MemoryMapAccess {
        Read = 0x0001,
        Write = 0x0002,
    };
    enum class Type {
        Vertex,
        Index,
        ShaderStorage,
        IndirectDraw,
    };
    enum class Usage {
        StreamDraw,
        StreamRead,
        StreamCopy,
        StaticDraw = 0x0004,
        StaticRead,
        StaticCopy,
        DynamicDraw = 0x0008,
        DynamicRead,
        DynamicCopy,
    };
    struct CreateInfo {
        uint32_t Size;
        Type BufferType;
        Usage BufferUsage;
        std::string_view DebugName;
    };
    struct MemoryUnmapper {
        void operator()(const uint8_t* mapped);
        uint32_t id_;
    };
    static std::unique_ptr<Buffer> create(const CreateInfo& info);
    virtual ~Buffer();

    std::unique_ptr<uint8_t, MemoryUnmapper> map(MemoryMapAccess access);
    void bind() const;
    void bind(Type type) const;

private:
    Buffer(uint32_t handle, uint32_t target, uint32_t size);

    const uint32_t handle_;
    const uint32_t target_;
    const uint32_t size_;

    friend class Mesh;
    friend class Pipeline;
    friend class ComputePipeline;
    friend class GraphicsPipeline;
};

#endif  // GRAPHICS_BUFFER_H
