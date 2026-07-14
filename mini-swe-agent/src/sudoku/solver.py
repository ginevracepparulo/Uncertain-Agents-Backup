def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None, None


def is_valid(board, num, pos):
    # Check row
    for c in range(9):
        if board[pos[0]][c] == num and pos[1] != c:
            return False

    # Check column
    for r in range(9):
        if board[r][pos[1]] == num and pos[0] != r:
            return False

    # Check 3x3 box
    box_r = pos[0] // 3
    box_c = pos[1] // 3

    for r in range(box_r * 3, box_r * 3 + 3):
        for c in range(box_c * 3, box_c * 3 + 3):
            if board[r][c] == num and (r, c) != pos:
                return False
    return True


def solve_sudoku(board):
    row, col = find_empty(board)

    if row is None:  # No empty cell found, board is solved
        return True

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # Backtrack

    return False
