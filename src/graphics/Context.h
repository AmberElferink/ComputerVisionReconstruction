#pragma once

#include <memory>
#include <string_view>

typedef void *SDL_GLContext;
struct SDL_Window;

/// RAII implementation of custom destructor of SDL objects so
/// they clean themselves up correctly when they go out of scope.
struct SDLDestroyer {
  void operator()(SDL_GLContext context) const;
  void operator()(SDL_Window *window) const;
};

/// Wrapper for OpenGL context and SDL window.
/// While this singleton lives, rendering will work
class Context {
public:
  /// Factory function. Returns null if there was an error
  static std::unique_ptr<Context> create(const std::string_view &title,
                                         uint32_t x, uint32_t y,
                                         uint32_t width, uint32_t height);

  virtual ~Context();

  /// Swap the backbuffer to screen so draw to the screen is done on new
  /// backbuffer
  void swapBuffers();

  /// Getter of the native window handle which is necessary for initializing
  /// the ui and other renderers.
  SDL_Window* getNativeWindowHandle() const;

  void getSize(int& width, int& height) const;
  float getAspectRatio() const;
private:
  /// Private unique constructor forcing the use of factory function which
  /// can return null unlike constructor.
  Context(SDL_Window *window, SDL_GLContext context);

  /// keeps track of SDL_Window and destroys it when it ceases to exist
  const std::unique_ptr<SDL_Window, SDLDestroyer> window_;
  const std::unique_ptr<void, SDLDestroyer> context_;
};
