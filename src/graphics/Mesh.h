#pragma once

#include <cstdint>

#include <memory>
#include <vector>
#include <string_view>

class Buffer;

/// Wrapper for OpenGL Vertex Array Buffers
class Mesh {
  public:
    enum Topology
    {
       Points = 0x0000,
       Lines = 0x0001,
       Line_loop = 0x0002,
       Line_strip = 0x0003,
       Triangles = 0x0004,
       Triangle_strip = 0x0005,
       Triangle_fan = 0x0006,
       Quads = 0x0007,
    };
    struct MeshAttributes
    {
      enum DataType
      {
        Int8 = 0x0000,
        Uint8 = 0x0001,
        Int16 = 0x0002,
        Uint16 = 0x0003,
        Int32 = 0x0004,
        Uint32 = 0x0005,
        Float = 0x0006,
      };
      DataType Type;
      uint32_t Count;
    };
    struct CreateInfo {
        const MeshAttributes* Attributes;
        uint32_t AttributeCount;
        Topology MeshTopology;
        std::string_view DebugName;
    };
    std::unique_ptr<Buffer> vertex_buffer_;
    std::unique_ptr<Buffer> index_buffer_;
    const uint32_t vao_;
    const uint32_t element_count_;
    const Topology topology_;

    /// General factory function only accepts uint32_t indices
    static std::unique_ptr<Mesh> create(const CreateInfo& info, std::unique_ptr<Buffer>&& vertex_buffer, std::unique_ptr<Buffer>&& index_buffer);
    /// Factory function for generating a full screen quad with positions
    /// encoded in the screen space coordinates
    static std::unique_ptr<Mesh>
    createFullscreenQuad(const std::string_view& debug_name);
    /// Factory function for creating a tri-color unit axis centered at 0
    /// with each arm extending at 1 in every axis.
    static std::unique_ptr<Mesh>
    createAxis(const std::string_view& debug_name);
    /// Factory function for creating a 1 unit cube with one vertex at the origin,
    /// every vertex with positive values in each axis and side length of 1.
    static std::unique_ptr<Mesh>
    createCube(const std::string_view& debug_name);

    virtual ~Mesh();
    /// Draw the indexed mesh using opengl
    void draw() const;
    void draw(uint32_t count) const;
    /// Bind the buffers for drawing
    void bind() const;

    Buffer& getVertexBuffer() const;
    Buffer& getIndexBuffer() const;

  private:
    /// Private unique constructor forcing the use of factory function which
    /// can return null unlike constructor.
    Mesh(std::unique_ptr<Buffer>&& vertex_buffer, std::unique_ptr<Buffer>&& index_buffer,
         uint32_t vao, uint32_t element_count, Topology topology);
};
