"""The board: a 5x5 grid, the four move directions and small position helpers.

Coordinates are (x, y). x grows to the east (right) and y grows to the SOUTH (down),
like rows on a page. So (0, 0) is the top-left cell and (4, 4) is the bottom-right.
A "layout" is a tuple of positions, one per object: layout[k] = (x, y) of object k.
We use tuples (not lists) so layouts can be stored in sets and dictionaries.
"""

GRID_SIZE = 5

# Each direction is a (dx, dy) step. Remember: south means y + 1.
DIRECTIONS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}
DIRECTION_NAMES = ["N", "S", "E", "W"]  # fixed order, used for action numbers


def in_bounds(cell):
    """True if the cell is on the 5x5 board."""
    x, y = cell
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def step(cell, direction):
    """The cell you reach by taking one step from `cell` in `direction` (may be off the board)."""
    dx, dy = DIRECTIONS[direction]
    return (cell[0] + dx, cell[1] + dy)


def manhattan(a, b):
    """Grid distance: how many N/S/E/W steps apart two cells are."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def moore_neighbors(cell):
    """The up-to-8 cells touching `cell`, including diagonals, that are on the board."""
    x, y = cell
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue  # the cell itself is not its own neighbor
            neighbor = (x + dx, y + dy)
            if in_bounds(neighbor):
                result.append(neighbor)
    return result


def move_object(layout, k, new_cell):
    """Return a NEW layout where object k sits at new_cell (layouts are never changed in place)."""
    as_list = list(layout)
    as_list[k] = new_cell
    return tuple(as_list)


def all_cells():
    """Every cell on the board, row by row."""
    return [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE)]
