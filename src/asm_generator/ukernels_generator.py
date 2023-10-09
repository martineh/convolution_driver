#!/usr/bin/python3

import argparse
import sys

import ARMv8
import RISCV
import common as cm

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def check_MR_NR(asm, arch):
    vr_max = asm.vr_max
    
    #Check Line Size integrity
    if (asm.mr % asm.vl != 0) or (asm.mr <= 0) or (asm.nr <= 0): return -1
    if (NR % asm.vl != 0): return -2
    if (arch == "riscv") and (asm.nr > len(asm.tmp_regs)): return -3

    #Register utilization
    MR_vregs  = asm.mr // asm.vl
    NR_vregs  = asm.maxB 

    if (arch == "riscv"):
        if (not asm.broadcast) and (not asm.gather):
            if ( NR_vregs >= len(asm.tmp_regs)):
                return -4;
            NR_vregs = 0;
        elif (asm.gather):
            NR_vregs += asm.nr // asm.vl

    C_vregs   = asm.mr // asm.vl * NR
    tot_vregs = MR_vregs + NR_vregs + C_vregs

    if asm.pipelining:
        if (arch == "riscv"):
            tot_vregs += MR_vregs
            if (asm.broadcast):
                tot_vregs += NR_vregs;
            elif(asm.gather):
                tot_vregs += asm.nr // asm.vl
        else:
            tot_vregs += MR_vregs + NR_vregs 
            tot_vregs += 2

    if tot_vregs > asm.vr_max: return -4
    
    return tot_vregs



if __name__ == "__main__":
    #Input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',       '-a', required=True, type=str, action='store', help="Architecture ASM code. Values=['riscv' | 'armv8']")
    parser.add_argument('--unroll',     '-u', required=False, type=int, help="Loop unrolling.")
    parser.add_argument('--pipelining', '-p', action='store_true', required=False, help="Enable Software pipelining. Warning: The number of the vector rigisters is incremented. A unroll x2 is applied")
    parser.add_argument('--reorder',    '-r', action='store_true', required=False, help="Reorder A|B load instructions.")
    parser.add_argument('--op_b',       '-o', required=False, type=str, action='store', help="B Matrix opetations method. Values=['broadcast' | 'gather']")

    args   = parser.parse_args()

    arch       = args.arch
    reorder    = args.reorder
    unroll     = args.unroll
    pipelining = args.pipelining
    broadcast  = False
    gather     = False
            
    if args.op_b is not None:
        if (arch != "riscv"):
            print ("Error: This option only is available for riscv.")
            sys.exit(-1)
        if (args.op_b == "broadcast"):
            broadcast = True
        elif (args.op_b == "gather"):
            gather = True
        else:
            print ("Error: Unknown B optimization. Values availables=['broadcast' | 'gather']")
            sys.exit(-1)
    if unroll is None:
        unroll = 0
    elif (pipelining):
        print ("Error: 'Unroll' and 'pipelining' options are incompatibles. 'Pipelining' option implies an unroll x2 factor.")
        sys.exit(-1)
    if (unroll % 2 != 0):
        print ("Error: Unroll must be multiple of two.")
        sys.exit(-1)
    if (broadcast) and (arch != "riscv"):
        print ("Error: Broadcast option only available with RISCV architecture.")
        sys.exit(-1)
    if (gather) and (arch != "riscv"):
        print ("Error: Gather option only available with RISCV architecture.")
        sys.exit(-1)
    if (reorder) and (arch != "riscv"):
        print ("Error: Reorder option only available with RISCV architecture.")
        sys.exit(-1)
    #if (arch == "riscv") and (pipelining): 
        #if (not broadcast) and (not gather):
        #    print ("Error: Riscv pipelining only supported with broadcast and gather option.")
        #    sys.exit(-1)
    if (arch != "riscv") and (arch != "armv8"):
        print ("Error: Architecture not supported.")
        sys.exit(-1)
            
    cm.clear_path()
    msg = ""
    print ("\n")
    print ("+============================================================+")
    print ("|                  MICRO-KERNELS GENERATOR                   |")
    print ("+============================================================+")
    print ("|    [*] ASM Architecture               | %-18s |" % (arch))
    if unroll == 0:
        msg = "Disable"
        if pipelining: msg = "x2 Factor"
    else:
        msg = "x%d Factor" % (unroll)
    print ("|    [*] Unroll                         | %-18s |" % (msg))
    if pipelining: 
        msg = "Enable" 
    else: msg = "Disable"

    print ("|    [*] Software pipelining            | %-18s |" % (msg))
    if (arch == "riscv") :
        if reorder: 
            msg = "Enable" 
        else: msg = "Disable"
        print ("|    [*] Reorder A-B Loads              | %-18s |" % (msg))
        
        if gather: 
            msg = "Enable" 
        else: msg = "Disable"
        print ("|    [*] Gather                         | %-18s |" % (msg))
        
        if broadcast: 
            msg = "Enable" 
        else: msg = "Disable"
        print ("|    [*] Broadcast                      | %-18s |" % (msg))
    print ("+=======================================+====================+")

    print ("")
    print ("+=================================================+=====+====+")
    print ("|               MICRO-KERNELS GENERATED           |  MR | NR |")
    print ("+=================================================+=====+====+")
    for mr in range(4, 24, 4):
        for nr in range(4, 24, 4):
            MR = mr
            NR = nr
            
            #Set configutation
            asm = None
            if arch == "riscv":
                asm = RISCV.ASM_RISCV(MR, NR, arch, broadcast, gather, reorder, unroll, pipelining)
            else:
                asm = ARMv8.ASM_ARMv8(MR, NR, arch, unroll, pipelining)
            
            tot_vregs = check_MR_NR(asm, arch)
            
            if tot_vregs > 0:
                #----------------------------------------------------------------
                #Generating micro-kernel
                #----------------------------------------------------------------
                asm.generate_umicro()
                cm.generate_edge_function(asm)
                cm.generate_selector_function(asm)
                print(f"|    [*] Micro-kernel                             | %s%-3d%s | %s%-3d%s|" % (bcolor.OKCYAN, MR, bcolor.ENDC, bcolor.OKCYAN, NR, bcolor.ENDC))
                #----------------------------------------------------------------
            
    cm.generate_selector_function(asm, close=True)
    print ("+=================================================+=====+====+\n")
