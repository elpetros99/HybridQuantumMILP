\ Model DC-OPF_2
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  11.67433 PG[0,0] + 10.34189 PG[0,1] + 10.84041 PG[0,2]
Subject To
 R0: - 5 delta[0,0] + 5 delta[0,1] + PL[0,0] + 10000 linestatus[0,0]
   <= 10000
 R1: - 5 delta[0,0] + 5 delta[0,3] + PL[0,1] + 10000 linestatus[0,1]
   <= 10000
 R2: - 3.333333333333333 delta[0,0] + 3.333333333333333 delta[0,4]
   + PL[0,2] + 10000 linestatus[0,2] <= 10000
 R3: - 4 delta[0,1] + 4 delta[0,2] + PL[0,3] + 10000 linestatus[0,3]
   <= 10000
 R4: - 10 delta[0,1] + 10 delta[0,3] + PL[0,4] + 10000 linestatus[0,4]
   <= 10000
 R5: - 3.333333333333333 delta[0,1] + 3.333333333333333 delta[0,4]
   + PL[0,5] + 10000 linestatus[0,5] <= 10000
 R6: - 5 delta[0,1] + 5 delta[0,5] + PL[0,6] + 10000 linestatus[0,6]
   <= 10000
 R7: - 3.846153846153846 delta[0,2] + 3.846153846153846 delta[0,4]
   + PL[0,7] + 10000 linestatus[0,7] <= 10000
 R8: - 10 delta[0,2] + 10 delta[0,5] + PL[0,8] + 10000 linestatus[0,8]
   <= 10000
 R9: - 2.5 delta[0,3] + 2.5 delta[0,4] + PL[0,9] + 10000 linestatus[0,9]
   <= 10000
 R10: - 3.333333333333333 delta[0,4] + 3.333333333333333 delta[0,5]
   + PL[0,10] + 10000 linestatus[0,10] <= 10000
 R11: 5 delta[0,0] - 5 delta[0,1] - PL[0,0] + 10000 linestatus[0,0]
   <= 10000
 R12: 5 delta[0,0] - 5 delta[0,3] - PL[0,1] + 10000 linestatus[0,1]
   <= 10000
 R13: 3.333333333333333 delta[0,0] - 3.333333333333333 delta[0,4] - PL[0,2]
   + 10000 linestatus[0,2] <= 10000
 R14: 4 delta[0,1] - 4 delta[0,2] - PL[0,3] + 10000 linestatus[0,3]
   <= 10000
 R15: 10 delta[0,1] - 10 delta[0,3] - PL[0,4] + 10000 linestatus[0,4]
   <= 10000
 R16: 3.333333333333333 delta[0,1] - 3.333333333333333 delta[0,4] - PL[0,5]
   + 10000 linestatus[0,5] <= 10000
 R17: 5 delta[0,1] - 5 delta[0,5] - PL[0,6] + 10000 linestatus[0,6]
   <= 10000
 R18: 3.846153846153846 delta[0,2] - 3.846153846153846 delta[0,4] - PL[0,7]
   + 10000 linestatus[0,7] <= 10000
 R19: 10 delta[0,2] - 10 delta[0,5] - PL[0,8] + 10000 linestatus[0,8]
   <= 10000
 R20: 2.5 delta[0,3] - 2.5 delta[0,4] - PL[0,9] + 10000 linestatus[0,9]
   <= 10000
 R21: 3.333333333333333 delta[0,4] - 3.333333333333333 delta[0,5]
   - PL[0,10] + 10000 linestatus[0,10] <= 10000
 R22: PG[0,0] <= 33.33333333335
 R23: PG[0,1] <= 25
 R24: PG[0,2] <= 30
 R25: - PG[0,0] <= 0
 R26: - PG[0,1] <= 0
 R27: - PG[0,2] <= 0
 R28: delta[0,2] <= 0
 R29: - delta[0,2] <= 0
 R30: delta[0,0] <= 360
 R31: delta[0,1] <= 360
 R32: delta[0,2] <= 360
 R33: delta[0,3] <= 360
 R34: delta[0,4] <= 360
 R35: delta[0,5] <= 360
 R36: - delta[0,0] <= 360
 R37: - delta[0,1] <= 360
 R38: - delta[0,2] <= 360
 R39: - delta[0,3] <= 360
 R40: - delta[0,4] <= 360
 R41: - delta[0,5] <= 360
 R42: - PL[0,0] - 40 linestatus[0,0] <= 0
 R43: - PL[0,1] - 60 linestatus[0,1] <= 0
 R44: - PL[0,2] - 40 linestatus[0,2] <= 0
 R45: - PL[0,3] - 40 linestatus[0,3] <= 0
 R46: - PL[0,4] - 60 linestatus[0,4] <= 0
 R47: - PL[0,5] - 30 linestatus[0,5] <= 0
 R48: - PL[0,6] - 90 linestatus[0,6] <= 0
 R49: - PL[0,7] - 70 linestatus[0,7] <= 0
 R50: - PL[0,8] - 80 linestatus[0,8] <= 0
 R51: - PL[0,9] - 20 linestatus[0,9] <= 0
 R52: - PL[0,10] - 40 linestatus[0,10] <= 0
 R53: PL[0,0] - 40 linestatus[0,0] <= 0
 R54: PL[0,1] - 60 linestatus[0,1] <= 0
 R55: PL[0,2] - 40 linestatus[0,2] <= 0
 R56: PL[0,3] - 40 linestatus[0,3] <= 0
 R57: PL[0,4] - 60 linestatus[0,4] <= 0
 R58: PL[0,5] - 30 linestatus[0,5] <= 0
 R59: PL[0,6] - 90 linestatus[0,6] <= 0
 R60: PL[0,7] - 70 linestatus[0,7] <= 0
 R61: PL[0,8] - 80 linestatus[0,8] <= 0
 R62: PL[0,9] - 20 linestatus[0,9] <= 0
 R63: PL[0,10] - 40 linestatus[0,10] <= 0
 R64: PG[0,0] - PL[0,0] - PL[0,1] - PL[0,2] <= 0
 R65: PG[0,1] + PL[0,0] - PL[0,3] - PL[0,4] - PL[0,5] - PL[0,6] <= 0
 R66: PG[0,2] + PL[0,3] - PL[0,7] - PL[0,8] <= 0
 R67: PL[0,1] + PL[0,4] - PL[0,9] <= 3.78
 R68: PL[0,2] + PL[0,5] + PL[0,7] + PL[0,9] - PL[0,10] <= 3.78
 R69: PL[0,6] + PL[0,8] + PL[0,10] <= 3.78
 R70: - PG[0,0] + PL[0,0] + PL[0,1] + PL[0,2] <= 0
 R71: - PG[0,1] - PL[0,0] + PL[0,3] + PL[0,4] + PL[0,5] + PL[0,6] <= 0
 R72: - PG[0,2] - PL[0,3] + PL[0,7] + PL[0,8] <= 0
 R73: - PL[0,1] - PL[0,4] + PL[0,9] <= -3.78
 R74: - PL[0,2] - PL[0,5] - PL[0,7] - PL[0,9] + PL[0,10] <= -3.78
 R75: - PL[0,6] - PL[0,8] - PL[0,10] <= -3.78
 R76: linestatus[0,0] + linestatus[0,1] + linestatus[0,2] + linestatus[0,3]
   + linestatus[0,4] + linestatus[0,5] + linestatus[0,6] + linestatus[0,7]
   + linestatus[0,8] + linestatus[0,9] + linestatus[0,10] <= 5
Bounds
 PG[0,0] <= 9999.9
 PG[0,1] <= 9999.9
 PG[0,2] <= 9999.9
 -360 <= delta[0,0] <= 360
 -360 <= delta[0,1] <= 360
 -360 <= delta[0,2] <= 360
 -360 <= delta[0,3] <= 360
 -360 <= delta[0,4] <= 360
 -360 <= delta[0,5] <= 360
 -999.9 <= PL[0,0] <= 999.9
 -999.9 <= PL[0,1] <= 999.9
 -999.9 <= PL[0,2] <= 999.9
 -999.9 <= PL[0,3] <= 999.9
 -999.9 <= PL[0,4] <= 999.9
 -999.9 <= PL[0,5] <= 999.9
 -999.9 <= PL[0,6] <= 999.9
 -999.9 <= PL[0,7] <= 999.9
 -999.9 <= PL[0,8] <= 999.9
 -999.9 <= PL[0,9] <= 999.9
 -999.9 <= PL[0,10] <= 999.9
Binaries
 linestatus[0,0] linestatus[0,1] linestatus[0,2] linestatus[0,3]
 linestatus[0,4] linestatus[0,5] linestatus[0,6] linestatus[0,7]
 linestatus[0,8] linestatus[0,9] linestatus[0,10]
End
