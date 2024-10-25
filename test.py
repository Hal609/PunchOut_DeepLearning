import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf

class GenericError(BaseException):pass

def create_debug_window(title="Debug Window", width=450, height=800):
    sdl2.ext.init()
    sdlttf.TTF_Init()
    font = sdlttf.TTF_OpenFont("droid_mono.ttf".encode('utf-8'), 20)
    
    if sdl2.ext.init() is not None:
        print("SDL2 initialization error:", sdl2.SDL_GetError().decode("utf-8"))
        raise InterruptedError
    if sdlttf.TTF_Init() == -1:
        print("SDL2_ttf initialization error:", sdl2.SDL_GetError().decode("utf-8"))
    if font is None:
        print("Failed to load font. SDL2 error:", sdl2.SDL_GetError().decode("utf-8"))
        return None
    
    # Create a window for debugging information
    window = sdl2.ext.Window(title, size=(width, height), position=(50, 50), flags=sdl2.SDL_WINDOW_ALLOW_HIGHDPI)
    window.show()

    # Create a renderer
    renderer = sdl2.ext.Renderer(window)
    renderer.logical_size = (width*2, height*2)
    
    return window, renderer, font

def show_text(renderer, text):
    # Set text color (white in this case)
    color = sdl2.SDL_Color(255, 255, 255)
    surface = sdlttf.TTF_RenderText_Blended(font, text.encode('utf-8'), color)
    # Get width and height of the text surface
    w, h = surface.contents.w, surface.contents.h

    # Create a destination rect for rendering, adjusting y position for each line
    dst_rect = sdl2.SDL_Rect(10, 10, w, h)

    texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, surface)
    if not texture:
        print("SDL2 error in SDL_CreateTextureFromSurface:", sdl2.SDL_GetError().decode("utf-8"))
    
    # Render texture to the renderer
    sdl2.SDL_RenderCopy(renderer.sdlrenderer, texture, None, dst_rect)

    # Clean up
    sdl2.SDL_FreeSurface(surface)
    sdl2.SDL_DestroyTexture(texture)

window, renderer, font = create_debug_window()
sdl2.SDL_RaiseWindow(window.window)

while True:
    events = sdl2.ext.get_events()
    for event in events:
        if event.type == sdl2.SDL_QUIT:
            sdl2.ext.quit()
            exit()
    renderer.clear(sdl2.ext.Color(0, 0, 0))
    show_text(renderer, "TEST")
    renderer.present()
