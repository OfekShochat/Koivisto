
/****************************************************************************************************
 *                                                                                                  *
 *                                     Koivisto UCI Chess engine                                    *
 *                                   by. Kim Kahre and Finn Eggers                                  *
 *                                                                                                  *
 *                 Koivisto is free software: you can redistribute it and/or modify                 *
 *               it under the terms of the GNU General Public License as published by               *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                    Koivisto is distributed in the hope that it will be useful,                   *
 *                  but WITHOUT ANY WARRANTY; without even the implied warranty of                  *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                           GNU General Public License for more details.                           *
 *                 You should have received a copy of the GNU General Public License                *
 *                 along with Koivisto.  If not, see <http://www.gnu.org/licenses/>.                *
 *                                                                                                  *
 ****************************************************************************************************/

#ifndef KOIVISTO_NEWMOVEGEN_H
#define KOIVISTO_NEWMOVEGEN_H

#include "Board.h"
#include "History.h"
#include "Move.h"
#include "Bitboard.h"

constexpr int MAX_QUIET = 128;
constexpr int MAX_NOISY = 32;

constexpr int MAX_HIST  = 512;

enum {
    PV_SEARCH,
    Q_SEARCH,
};

enum {
    GET_HASHMOVE,
    GEN_NOISY,
    GET_GOOD_NOISY,
    GEN_QUIET,
    GET_QUIET,
    GET_BAD_NOISY,
    END,
};

class moveGen {
    private:

    int             stage;

    Move            quiets[MAX_QUIET];
    Move            noisy[MAX_NOISY];
    Move            searched[MAX_QUIET];
    int             quietScores[MAX_QUIET];
    int             noisyScores[MAX_NOISY];
    int             quietSize;
    int             noisySize;
    int             goodNoisyCount;
    int             noisy_index;
    int             quiet_index;
    int             searched_index;

    Board*          m_board;
    SearchData*     m_sd;
    Depth           m_ply;
    Move            m_hashMove;
    int             m_mode;

    public: 
    void init(SearchData* sd, Board* b, Depth ply, Move hashMove, int mode);
    Move next();
    void addNoisy(Move m);
    void addQuiet(Move m);
    Move nextNoisy();
    Move nextQuiet();
    void addSearched(Move m);
    void generateNoisy();
    void generateQuiet();
    void updateHistory(int weight, Move previous, Move followup);
};

#endif