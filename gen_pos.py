import chess
import chess.syzygy
import random
import numpy as np


def gen_fen(material):
    """Return a board with the specific material balance"""""
    # Step 1: Create a blank board with either white or black to move
    # We are assuming no castling or ep capture will be possible


    # Step 2: Decide whether the first group of material is for white or black
    if random.randint(0, 1) == 1:
        material = material.swapcase()

    # Step 3: Loop over material and add it to the board
    while True:
        if random.randint(0, 1) == 1:
            board = chess.Board('8/8/8/8/8/8/8/8 w - - 0 1')
        else:
            board = chess.Board('8/8/8/8/8/8/8/8 b - - 0 1')
        dest_squares = random.sample(range(64), len(material))
        for piece in material:
            location = dest_squares.pop()
            pc = chess.Piece.from_symbol(piece)
            board.set_piece_at(location, pc)
		# Step 4: Check that the position is valid
        if board.is_valid() and not board.is_checkmate() and not board.is_stalemate():
			# Recursively get a new try
            return board

def is_white_square(square):
    """Returns True if the square is on the 'white' diagonals"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if (file + rank) % 2 == 0:
        return 1
    else:
        return 0


def board_to_plane(board):
    """Takes a board position and translates it into a plane of binary values"""
    """Old code that returned a vector"""
    # 0-63,448-511: bishop on dark squares    PNBRQK == 123456
    # 64- 512-:  black pawns, white pawns     white = True, black = False
    # 128- 576-: knight
    # 192- 640-: bishop on light squares
    # 256- 704-: rook
    # 320- 768-: queen
    # 384- 832-895: king
    # The location of a square is described by:
    # 448*(color white=1 black=0) + 64*(piecetype 1-6) + square
    #     - 64*3*is_a_bishop*is_on_light_square
    # 896-961 : side to move color (white = 1s)
    plane = np.zeros(896, dtype=int)
    for square in range(64):
        piece_type = board.piece_type_at(square)
        if piece_type is None:
            pass
        else:
            if board.color_at(square) == chess.WHITE:
                piece_color = 1
            else:
                piece_color = 0
            index = 448 * piece_color + 64 * piece_type + square
            if piece_type == chess.BISHOP:
                # Bishops on white diagonals go in the 0 index instead of 3.
                index = index - 64 * 3 * is_white_square(square)
            plane[index] = 1
    if board.turn == chess.WHITE:
        col = np.ones(64)
    else:
        col = np.zeros(64)
    # print(plane)
    plane2 = np.concatenate((plane, col))
    return plane2


def board_to_planev1(board):
    """Takes a board position and translates it into a planes of binary values"""
    """New code retains the 8x8 planes"""
    # The resulting plane will be (8,8,15) - apparently tensorflow prefers channels last
    # 0, 7: bishop on dark squares    PNBRQK == 123456
    # 1- 8-:  black pawns, white pawns     white = True, black = False
    # 2- 9-: knight
    # 3- 10-: bishop on light squares
    # 4- 11-: rook
    # 5- 12-: queen
    # 6- 13 : king
    # 14    : color white = 1
    # The location of a square is described by:
    # 448*(color white=1 black=0) + 64*(piecetype 1-6) + square
    #     - 64*3*is_a_bishop*is_on_light_square
    # 896-961 : side to move color (white = 1s)
    plane = np.zeros((8, 8, 15), dtype=int)
    for square in range(64):
        piece_type = board.piece_type_at(square)
        if piece_type is None:
            pass
        else:
            if board.color_at(square) == chess.WHITE:
                piece_color = 1
            else:
                piece_color = 0
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            index = 7 * piece_color + piece_type
            if piece_type == chess.BISHOP:
                # Bishops on white diagonals go in the 0/7 index instead of 3/10.
                index = index - 3 * is_white_square(square)

            plane[rank, file, index] = 1
    if board.turn == chess.WHITE:
        col = 1
    else:
        col = 0
    plane[:, :, 14] = col
    return plane


def board_label(board, tablebase, f):
    """Returns the training labels for the board from Syzygy lookup"""
    # 0 draw for side-to-move, 1 win for side-to-move (more than 50 moves), 2 win for side-to-move
    # -1 loss in more than 50, -2 loss in <50
        # board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
    wdl = tablebase.probe_wdl(board)

    # 0 draw, x win in x, -x loss in x
    # counts may be off by 1
   
        # board = chess.Board("8/2K5/4B3/3N4/8/8/4k3/8 b - - 0 1")
    dtz = tablebase.probe_dtz(board)
    if wdl == 0:
        win = 0
        draw = 1
        loss = 0
    elif wdl > 0:
        win = 1
        draw = 0
        loss = 0
    else:
        win = 0
        draw = 0
        loss = 1
    if dtz > 0:
        quality = 2000 - dtz
    elif dtz < 0:
        quality = -2000 - dtz
    else:
        quality = 0
    print(f"{board.fen()}|{dtz}|{float(win + (draw/2) - loss):.1f}", file=f)
    return win, draw, loss, quality


def adjust_case(input_str):
    """This converts endgame descriptors so that the first block is capitalized"""
    """and the second block is lowercase.  e.g.  krpkq to KRPkq"""
    lower = input_str.lower()
    second_k = lower.find("k", 1)
    # print(f"second k at {second_k}")
    out1 = lower[:second_k].upper()
    out2 = lower[second_k:]
    output_str = out1+out2
    if second_k == -1:
        output_str = "fail"
    return output_str


def ask_for_input():
    need_input = True
    while need_input:
        balance = input("Please enter the endgame (e.g. KRkp): ")
        balance = adjust_case(balance)
        # print(f"After Adjusting: {balance}")
        number = int(input("Please enter the number of training examples: "))
        if number > 0:
            need_input = False
        if balance == "fail":
            need_input = True
        if need_input:
            print("There was a problem with the inputs.")

    return balance, number

from tqdm import tqdm
def generate_training():
    material_balance, target_count = ask_for_input()
    plane_version = 'v1'
    epdfile = "positions.epd"
    # Note: it took about 1:30 to generate 10,000 positions
    X_train = []
    y_train = []
    with chess.syzygy.open_tablebase("/Volumes/Samsung_T5/egdb") as tablebase:
        with open('positions.epd', 'a') as f:    
            for i in tqdm(range(target_count), desc="Generating...", ascii=False, ncols=75):
                my_board = gen_fen(material_balance)
                my_plane = board_to_planev1(my_board)
                my_label = board_label(my_board, tablebase, f)
                X_train.append(my_plane)
                y_train.append(my_label)
                #print(i)

        # Converts from a list to an array at the end; faster than concat array
        X_train = np.stack(X_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        print(X_train.shape)
        print(y_train.shape)
        print("frequency list:")
        unique_elements, counts_elements = np.unique(y_train[:, 3], return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))

        outfile = "./training/train_"+material_balance+str(int(target_count/1000))+"K"+plane_version+".npz"
        # Save as a compressed npz file
        np.savez_compressed(outfile, X_train=X_train, y_train=y_train)
        print(f"Data saved to {outfile}")

    # test that we can read the data
    # print("Doing a test read of the data")
    # npzfile2 = np.load(outfile)
    #    print("npz variables")
    #    print(npzfile.files)
    #    print(npzfile['y_train'])
    # X_t2 = npzfile2['X_train']
    # y_t2 = npzfile2['y_train']
    # print(X_t2.shape)
    # print(y_t2.shape)
    #print(y_t2)
if __name__ == '__main__':
    yes_no = input("Do you need to generate endgame training data? ")
    if len(yes_no) == 0:
        pass
    elif yes_no[0].lower() == "y":
        generate_training()
