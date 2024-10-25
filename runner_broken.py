import sdl2
import sdl2.ext
import sdl2.sdlttf as sdlttf
from cynes.windowed import WindowedNES
from cynes import NES
from cynes import *

import signal
import sys

import time
from utils import *

from numpy.typing import NDArray
import numpy as np
from typing import Optional

class NESWindow():
    def __init__(self, rom_path, headless=False, show_debug=True):
        self.rom_path = rom_path
        self.font_path = "droid_mono.ttf"
        self.headless = headless
        self.show_debug = show_debug
        
        self.nes = None
        self.frame:Optional[NDArray[np.uint8]]

        self.running = False
        self.frame_rate = 0
        self.sdl_event = sdl2.SDL_Event()
        self.sdl_renderer = None
        self.window = None
        self.debug_font = None

        # Register cleanup function for common exit signals
        signal.signal(signal.SIGINT, self.clean_up)
        signal.signal(signal.SIGTERM, self.clean_up)

        self.reward_view_string = ""
        self.training_epsilon = 1.0

        self.debug_text = ""
        self.debug_live_text = ""
        self.last_debug_update = 9999
        
        self.ram_watches = []
        self.inputs = {}

    def add_ram_watch(self, label: str, address: int):
        self.ram_watches.append((label, address))

    def format_text(self):
        self.debug_text = f"fps = {self.frame_rate:.0f}\n\n"
        print("Debug text:", self.debug_text)

        self.debug_text += f"INPUT\n====="
        self.debug_text += str(self.inputs)

        self.debug_text += f"\n\nREWARDS\n======="
        self.debug_text += self.reward_view_string

        self.debug_text += f"\n\nEPSILON\n=======\n"
        self.debug_text += str(self.training_epsilon)

        self.debug_text += "\n\nRAM WATCH\n=========\n"
        for label, address in self.ram_watches:
            self.debug_text += f"{label}: {self.nes[address]}\n"

        self.debug_text += "\n\nMESSAGES\n========"
        self.debug_text += self.debug_live_text

    def debug_print(self, text: str, clear=False):
        if clear:
            self.debug_live_text = "\n" + text
            return
        
        line_limit = 25
        if self.debug_live_text.count("\n") > line_limit:
            self.debug_live_text = self.debug_live_text[self.debug_live_text.find("\n", 1):]
        self.debug_live_text += "\n" + text
    
    def create_debug_window(self, title="Debug Window", width=450, height=800):
        sdl2.ext.init()
        sdlttf.TTF_Init()
        if sdl2.ext.init() is not None:
            raise GenericError(f"SDL2 initialization error: {sdl2.SDL_GetError().decode("utf-8")}")
        if sdlttf.TTF_Init() == -1:
            raise GenericError(f"SDL2_ttf initialization error: {sdl2.SDL_GetError().decode("utf-8")}")
        
        # Load a font for rendering text
        font = sdlttf.TTF_OpenFont(self.font_path.encode('utf-8'), 20)

        if font is None:
            raise GenericError(f"Failed to load font. SDL2 error: {sdl2.SDL_GetError().decode("utf-8")}")
        
        # Create a window for debugging information
        window = sdl2.ext.Window(title, size=(width, height), position=(50, 50), flags=sdl2.SDL_WINDOW_ALLOW_HIGHDPI)
        window.show()

        # Create a renderer
        renderer = sdl2.ext.Renderer(window)
        renderer.logical_size = (width*2, height*2)
        
        return window, renderer, font

    def render_text(self, renderer, font, text, x, y, line_height=20, max_length=75):
        def wrap_text(text, max_chars=39):
            """
            Wrap the text so that no line exceeds max_chars characters,
            while also supporting manual line breaks using '\n'.
            """
            # Split the text by manual line breaks (\n)
            lines = text.split('\n')
            wrapped_lines = []

            for line in lines:
                words = line.split(' ')
                current_line = []

                for word in words:
                    # Check if adding the next word would exceed the max character limit
                    if len(' '.join(current_line + [word])) <= max_chars:
                        current_line.append(word)
                    else:
                        # If the current line length exceeds max_chars, join the current_line
                        # and start a new one
                        wrapped_lines.append(' '.join(current_line))
                        current_line = [word]

                # Append any remaining words in the current_line as the last line for this section
                if current_line:
                    wrapped_lines.append(' '.join(current_line))

            return wrapped_lines
        
        if text == "": return
                
        # Wrap the text to ensure each line fits within max_chars
        wrapped_lines = wrap_text(text, max_length)

        # Set text color (white in this case)
        color = sdl2.SDL_Color(255, 255, 255)

        # Iterate over each line and render it at an increasing y position
        for i, line in enumerate(wrapped_lines):
            if line == "":
                continue
            # Render the line as surface using Blended method for better quality
            surface = sdlttf.TTF_RenderText_Blended(font, line.encode('utf-8'), color)

            if not surface:
                print("SDL2_ttf error in TTF_RenderText_Blended:", sdl2.SDL_GetError().decode("utf-8"))
                continue

            # Convert surface to texture
            texture = sdl2.SDL_CreateTextureFromSurface(renderer.sdlrenderer, surface)
            if not texture:
                print("SDL2 error in SDL_CreateTextureFromSurface:", sdl2.SDL_GetError().decode("utf-8"))
                continue

            # Get width and height of the text surface
            w, h = surface.contents.w, surface.contents.h

            # Create a destination rect for rendering, adjusting y position for each line
            dst_rect = sdl2.SDL_Rect(x, y + i * line_height, w, h)

            # Render texture to the renderer
            sdl2.SDL_RenderCopy(renderer.sdlrenderer, texture, None, dst_rect)

            # Clean up
            sdl2.SDL_FreeSurface(surface)
            sdl2.SDL_DestroyTexture(texture)

    def setup(self):
         # Create the debug window
        if self.show_debug:
            self.window, self.sdl_renderer, self.debug_font = self.create_debug_window()

            ram_dict = ram_dict_from_csv("ram_data.csv")

            for key in ram_dict.keys():
                self.add_ram_watch(key, ram_dict[key])
        
        self.running = True
        self.nes = NES(self.rom_path) if self.headless else WindowedNES(self.rom_path)

        if self.show_debug: sdl2.SDL_RaiseWindow(self.window.window)

    def clean_up(self):
        print("Cleaning up SDL resources...")
        if self.debug_font:
            sdlttf.TTF_CloseFont(self.debug_font)
        if self.sdl_renderer:
            self.sdl_renderer.destroy()
        if self.window:
            self.window.close()
        sdlttf.TTF_Quit()
        sdl2.ext.quit()
        print("Cleanup completed.")
        sys.exit(0)

    def perform_inputs(self):
        # Convert the dictionary back into an integer 
        self.nes.controller = sum(val << i for i, val in enumerate(self.inputs.values()))

    def run(self):
        self.setup()

        while self.running and not self.nes.should_close:
            self.step()

        self.clean_up()

    def step(self):
        if self.show_debug: 
            start_time = time.perf_counter()  # Start timing

            self.perform_inputs()
            
            self.frame = self.nes.step(frames=1)

            while sdl2.SDL_PollEvent(self.sdl_event) != 0:
                if self.sdl_event.type == sdl2.SDL_QUIT:
                    self.running = False
                    self.clean_up()
                    return
                
            current_time = time.perf_counter()
            if current_time - self.last_debug_update >= 0.05:

                # Clear the renderer with a black background
                self.sdl_renderer.clear(sdl2.ext.Color(0, 0, 0))

                self.format_text()
                # Render the memory address and value as text
                self.render_text(self.sdl_renderer, self.debug_font, self.debug_text, 10, 10)
                self.last_debug_update = current_time  # Reset the last update time

                # Present the renderer to update the window
                self.sdl_renderer.present()

            end_time = time.perf_counter()  # End timing

            # Calculate time taken for the loop iteration
            elapsed_time = end_time - start_time

            self.frame_rate = 1 / elapsed_time

        else:
            self.perform_inputs()
            self.frame = self.nes.step(frames=1)