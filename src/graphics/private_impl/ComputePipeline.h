#ifndef COMPUTEPIPELINE_H
#define COMPUTEPIPELINE_H

#include "../Pipeline.h"

class ComputePipeline : public Pipeline
{
public:
  /// Private unique constructor forcing the use of factory function which
  /// can return null unlike constructor.
  explicit ComputePipeline(uint32_t program);
  virtual ~ComputePipeline();

  void bind() override;
  bool setUniform(const std::string_view& uniform_name, const glm::vec3& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const glm::vec4& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const glm::mat4& uniform) override;
  bool setUniform(const std::string_view& uniform_name, const float& uniform) override;
  bool setUniform(const std::string_view& uniform_name, uint32_t index, const Buffer& uniform) override;

  const uint32_t program_;
};

#endif // COMPUTEPIPELINE_H
