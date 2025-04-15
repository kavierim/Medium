import pygame
import math
import random # Import the random module

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BACKGROUND_COLOR = (0, 0, 20) # Dark Blue
HIGHLIGHT_COLOR = (255, 255, 0, 150) # Yellow, semi-transparent
GRID_LINE_COLOR = (50, 50, 50) # Dark grey for grid lines

# Define two different tile colors
TILE_COLOR_A = (0, 150, 0) # Green
TILE_COLOR_B = (0, 100, 150) # Bluish-Green

# Isometric Tile Dimensions (using 2:1 ratio)
TILE_WIDTH = 128
TILE_HEIGHT = TILE_WIDTH // 2
TILE_WIDTH_HALF = TILE_WIDTH // 2
TILE_HEIGHT_HALF = TILE_HEIGHT // 2

# Map Dimensions (Example: 10x10 grid)
MAP_WIDTH = 10
MAP_HEIGHT = 10

# Screen offset
SCREEN_OFFSET_X = (SCREEN_WIDTH - (MAP_WIDTH + MAP_HEIGHT) * TILE_WIDTH_HALF / 2) / 2 + (MAP_WIDTH * TILE_WIDTH_HALF / 2)
SCREEN_OFFSET_Y = 100
SCREEN_OFFSET = pygame.math.Vector2(SCREEN_OFFSET_X, SCREEN_OFFSET_Y)

# --- Coordinate Transformation Functions (Unchanged) ---
def map_to_screen(map_pos):
    screen_x = (map_pos.x - map_pos.y) * TILE_WIDTH_HALF
    screen_y = (map_pos.x + map_pos.y) * TILE_HEIGHT_HALF
    screen_pos = pygame.math.Vector2(screen_x, screen_y) + SCREEN_OFFSET
    return screen_pos

def screen_to_map(screen_pos):
    adjusted_screen_pos = screen_pos - SCREEN_OFFSET
    map_x_float = (adjusted_screen_pos.x / TILE_WIDTH_HALF + adjusted_screen_pos.y / TILE_HEIGHT_HALF) / 2
    map_y_float = (adjusted_screen_pos.y / TILE_HEIGHT_HALF - adjusted_screen_pos.x / TILE_WIDTH_HALF) / 2
    return pygame.math.Vector2(map_x_float, map_y_float)

# --- Helper functions for drawing polygons (Reverted to this method) ---
def draw_iso_tile_outline(surface, screen_pos, color, line_width=1):
    """Draws the diamond shape outline of an isometric tile."""
    points = [
        screen_pos,
        screen_pos + pygame.math.Vector2(TILE_WIDTH_HALF, TILE_HEIGHT_HALF),
        screen_pos + pygame.math.Vector2(0, TILE_HEIGHT),
        screen_pos + pygame.math.Vector2(-TILE_WIDTH_HALF, TILE_HEIGHT_HALF)
    ]
    pygame.draw.lines(surface, color, True, points, line_width)

def draw_iso_tile_filled(surface, screen_pos, color):
    """Draws the filled diamond shape of an isometric tile."""
    points = [
        screen_pos,
        screen_pos + pygame.math.Vector2(TILE_WIDTH_HALF, TILE_HEIGHT_HALF),
        screen_pos + pygame.math.Vector2(0, TILE_HEIGHT),
        screen_pos + pygame.math.Vector2(-TILE_WIDTH_HALF, TILE_HEIGHT_HALF)
    ]
    pygame.draw.polygon(surface, color, points)

# --- Main Game Setup ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Isometric Demo - Random Colors") # Updated caption
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

# --- Create Map Data with Random Colors ---
map_data = {}
for y in range(MAP_HEIGHT):
    for x in range(MAP_WIDTH):
        # Randomly choose one of the two colors for this tile
        chosen_color = random.choice([TILE_COLOR_A, TILE_COLOR_B])
        # Store the chosen color in the map data dictionary
        map_data[(x, y)] = {'color': chosen_color}

# Store map coordinates for easier iteration and sorting
map_coords_list = list(map_data.keys())
map_coords_list.sort(key=lambda coord: (coord[1], coord[0])) # Simple depth sort

mouse_screen_pos = pygame.math.Vector2(0, 0)
selected_map_tile = None

# --- Game Loop ---
running = True
while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEMOTION:
            mouse_screen_pos = pygame.math.Vector2(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            map_float = screen_to_map(mouse_screen_pos)
            clicked_tile_coord = (math.floor(map_float.x), math.floor(map_float.y))
            # Check if the click is within the map bounds
            if 0 <= clicked_tile_coord[0] < MAP_WIDTH and 0 <= clicked_tile_coord[1] < MAP_HEIGHT:
                # *** Retrieve and print the color of the clicked tile ***
                clicked_tile_info = map_data.get(clicked_tile_coord)
                if clicked_tile_info:
                    clicked_tile_color = clicked_tile_info['color']
                    print(f"Clicked Tile: {clicked_tile_coord}, Stored Color: {clicked_tile_color}")
                else: # Should not happen if bounds check passes, but good practice
                     print(f"Clicked Tile: {clicked_tile_coord}, but no data found?")
            else:
                print(f"Clicked outside map area.")

    # --- Update Logic ---
    current_hover_map_float = screen_to_map(mouse_screen_pos)
    current_hover_map_tile = (math.floor(current_hover_map_float.x), math.floor(current_hover_map_float.y))

    if 0 <= current_hover_map_tile[0] < MAP_WIDTH and 0 <= current_hover_map_tile[1] < MAP_HEIGHT:
        selected_map_tile = current_hover_map_tile
    else:
        selected_map_tile = None

    # --- Drawing ---
    screen.fill(BACKGROUND_COLOR)

    # Draw the map tiles using stored colors
    for map_x, map_y in map_coords_list:
        map_pos = pygame.math.Vector2(map_x, map_y)
        screen_pos = map_to_screen(map_pos)

        # *** Get the color for this specific tile from map_data ***
        tile_info = map_data.get((map_x, map_y))
        if tile_info:
            tile_color = tile_info['color']
            # Draw the filled tile with its assigned color
            draw_iso_tile_filled(screen, screen_pos, tile_color)
            # Draw a thin outline for definition
            draw_iso_tile_outline(screen, screen_pos, GRID_LINE_COLOR, 1)
        else:
             # Fallback if data missing (shouldn't happen here)
             draw_iso_tile_outline(screen, screen_pos, (255,0,0), 2)


    # Highlight the selected tile
    if selected_map_tile:
        selected_screen_pos = map_to_screen(pygame.math.Vector2(selected_map_tile[0], selected_map_tile[1]))
        # Use outline for highlight over polygons
        draw_iso_tile_outline(screen, selected_screen_pos, (255, 255, 0), 3) # Thicker yellow outline


    # Display coordinates (Unchanged)
    mouse_text = font.render(f"Mouse Screen: ({int(mouse_screen_pos.x)}, {int(mouse_screen_pos.y)})", True, (255, 255, 255))
    map_text_float = font.render(f"Calculated Map (Float): ({current_hover_map_float.x:.2f}, {current_hover_map_float.y:.2f})", True, (255, 255, 255))
    if selected_map_tile:
        map_text_int = font.render(f"Hovered Map Tile: {selected_map_tile}", True, (255, 255, 255))
    else:
         map_text_int = font.render(f"Hovered Map Tile: (Outside Map)", True, (255, 255, 255))
    screen.blit(mouse_text, (10, 10))
    screen.blit(map_text_float, (10, 30))
    screen.blit(map_text_int, (10, 50))

    # --- Update Display ---
    pygame.display.flip()
    clock.tick(60)

# --- Clean Up ---
pygame.quit()