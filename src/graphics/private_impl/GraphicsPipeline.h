#ifndef GRAPHICSPIPELINE_H
#define GRAPHICSPIPELINE_H

#include "../Pipeline.h"

class GraphicsPipeline : public Pipeline
{
public:
  /// Private unique constructor forcing the use of factory function which
  /// can return null unlike constructor.
  GraphicsPipeline(uint32_t program, uint32_t viewportWidth, uint32_t viewportHeight, float lineWidth);
  virtual ~GraphicsPipeline();

  void bind() override;
  bool setUniform(const std::string_view& uniform_name, const glm::vec3& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const glm::vec4& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const glm::mat4& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const float& uniform) override;
  bool setUniform(const std::string_view& uniform_name, uint32_t index, const Buffer& uniform) override;

  const uint32_t program_;
  const uint32_t viewportWidth_;
  const uint32_t viewportHeight_;
  const float lineWidth_;
};

#endif // GRAPHICSPIPELINE_H
