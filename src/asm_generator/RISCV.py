import common as cm

#=====================================================================
#                           MODULE ARCH #0                           |
#=====================================================================
#|          Architecture Configuration          |       RISC-V       |
#=====================================================================
class ASM_RISCV():

    def __init__(self, MR, NR, arch, broadcast, gather, reorder, unroll, pipelining, tag=""):
        self.tag         = tag
        self.edge        = False
        self.vl          = 4  #Register line size
        self.vr_max      = 32 #Maximum number of vectorial registers
        self.data_type   = 4
        self.arch        = arch
        self.unroll      = unroll
        self.pipelining  = pipelining

        if pipelining:
            self.unroll  = 2
        else:
            self.unroll  = unroll

        self.maxB      = NR
        self.orig_mr   = MR
        self.orig_nr   = NR
        self.mr        = MR
        self.nr        = NR
        
        self.a_dsp       = (MR // self.vl) * (self.data_type * self.vl) #=MR * self.data_type
        self.b_dsp       = (NR // self.vl) * (self.data_type * self.vl) #=NR * self.data_type

        self.broadcast = broadcast
        self.gather    = gather
        self.reorder   = reorder

        self.r_kc      = "a0"
        self.r_alpha   = "a1"
        self.r_a       = "a2"
        self.r_b       = "a3"
        self.r_beta    = "a4"
        self.r_c       = "a5"
        self.r_ldC     = "a6"
        self.r_unroll  = "t0"
        self.r_kc_iter = "t1"
        self.r_kc_left = "t2"
    
        self.regs   = ["t0",  "t1",  "t2",  "t3",  "t4",  "t5",  "t6",  "s0", 
                       "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",  "s8", 
                       "s9",  "s10", "s11"]
    
        self.v_regs = ["v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",  "v8",  "v9",
                       "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                       "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                       "v30", "v31"]
    
        self.tmp_regs = ["ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
                         "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7"]
        
        self.AB_vreg = 0

        #Micro-Kernel Output Files
        self.fout = None
        cm.clear_path()

    def set_edge(self, new_MR, new_NR):
        if  (new_MR != 0) and (new_NR != 0) and (new_MR % self.vl == 0) :
            self.edge = True
            self.mr   = new_MR
            self.nr   = new_NR
            return 1
        else:
            return -1

    def __set_output(self):
        cm.add_header(self.mr, self.nr)
        self.fout = cm.new_micro_file(self.mr, self.nr)
    
    
    def __header(self):
        self.fout.write(f"  .text\n")
        self.fout.write(f"  .align      2\n")
        self.fout.write(f"  .global     gemm_ukernel_asm_{self.mr}x{self.nr}\n")
        self.fout.write(f"  //void gemm_ukernel_asm(size_t kc, float *alpha, float *a, float *b, float *beta, float *c, size_t ldC); \n")
        self.fout.write(f"    // register arguments:\n")
        self.fout.write(f"    // a0   kc\n")
        self.fout.write(f"    // a1   alpha\n")
        self.fout.write(f"    // a2   a\n")
        self.fout.write(f"    // a3   b\n")
        self.fout.write(f"    // a4   beta\n")
        self.fout.write(f"    // a5   c\n")
        self.fout.write(f"    // a6   ldC\n")
    
    
    def __registers_to_macros(self):
        MR = self.mr
        NR = self.nr
        #Input Parameters -RiscV-
        self.fout.write(f"  #define kc         {self.r_kc}\n")
        self.fout.write(f"  #define alpha_ptr  {self.r_alpha}\n")
        self.fout.write(f"  #define A0_ptr     {self.r_a}\n")
        self.fout.write(f"  #define B0_ptr     {self.r_b}\n")
        self.fout.write(f"  #define beta_ptr   {self.r_beta}\n")
        self.fout.write(f"  #define C00_ptr    {self.r_c}\n")
        self.fout.write(f"  #define ldC        {self.r_ldC}\n")
        self.fout.write(f"  #define unroll     {self.r_unroll}\n")
        self.fout.write(f"  #define kc_iter    {self.r_kc_iter}\n")
        self.fout.write(f"  #define kc_left    {self.r_kc_left}\n")
        self.fout.write(f"\n")
        
        reg    = 3
        vreg   = 0
        tmpreg = 0

        #-------------------------------------------------------------------
        #C with ldC strides address storage - A & B Loads addresses
        #-------------------------------------------------------------------
        for c in range(1, self.nr):
            self.fout.write(f"  #define C0{c}_ptr    {self.regs[reg]}\n")
            reg += 1
        self.fout.write(f"\n")

        #A to vector registers
        for a in range(1, self.mr // self.vl):
            self.fout.write(f"  #define A{a}_ptr     {self.regs[reg]}\n")
            reg += 1
        self.fout.write(f"\n")

        #B to vector registers
        for b in range(1, self.nr // self.vl):
            self.fout.write(f"  #define B{b}_ptr     {self.regs[reg]}\n")
            reg += 1
        self.fout.write(f"\n")
        #-------------------------------------------------------------------
       
        #-------------------------------------------------------------------
        # Vectorial Register Management
        #-------------------------------------------------------------------
        for r in range(0, self.mr // self.vl):
            for c in range(0, NR):
                self.fout.write(f"  #define C{r}{c}        {self.v_regs[vreg]}\n")
                vreg += 1
        self.fout.write(f"\n")
        
        #A to vector registers
        for r in range(0, self.mr // self.vl):
            self.fout.write(f"  #define A{r}         {self.v_regs[vreg]}\n")
            vreg += 1
        self.fout.write(f"\n")
   
        #B to vector registers
        if (self.broadcast) or (self.gather):
            for r in range(0, self.maxB):
                self.fout.write(f"  #define B{r}         {self.v_regs[vreg]}\n")
                vreg += 1
            self.fout.write(f"\n")

        if self.gather:
            #B to vector registers
            for r in range(0, self.nr // self.vl):
                self.fout.write(f"  #define BTMP{r}      {self.v_regs[vreg]}\n")
                vreg += 1
            self.fout.write(f"\n")

        if self.pipelining and (not self.edge):
            #A to vector registers
            for r in range(0, self.mr // self.vl):
                self.fout.write(f"  #define A{r}_prfm    {self.v_regs[vreg]}\n")
                vreg += 1
            self.fout.write(f"\n")

            if self.gather:
                #B to vector registers
                for r in range(0, self.nr // self.vl):
                    self.fout.write(f"  #define BTMP{r}_prfm    {self.v_regs[vreg]}\n")
                    vreg += 1
                self.fout.write(f"\n")
            else:
                #B to vector registers
                for r in range(0, self.maxB):
                    self.fout.write(f"  #define B{r}_prfm    {self.v_regs[vreg]}\n")
                    vreg += 1
                self.fout.write(f"\n")
        #-------------------------------------------------------------------


        #-------------------------------------------------------------------
        #Temporal registers management
        #-------------------------------------------------------------------
        for r in range(0, self.maxB):
            self.fout.write(f"  #define BF{r}        {self.tmp_regs[tmpreg]}\n")
            tmpreg += 1
        self.fout.write(f"\n")
        #-------------------------------------------------------------------
        self.fout.write(f"  #define TMP        {self.regs[reg]}\n")
        self.fout.write(f"  #define BETA       fa1\n")
        self.fout.write(f"\n")
   
        self.fout.write(f"  //ASM Instructions\n")
        self.fout.write(f"  #define LOAD       vle32.v\n")
        self.fout.write(f"  #define STORE      vse32.v\n")
        self.fout.write(f"  #define GATHER     vrgather.vi\n")
        if (not self.broadcast) and (not self.gather):
            self.fout.write(f"  #define VFMACC     vfmacc.vf\n")
        else:
            self.fout.write(f"  #define VFMACC     vfmacc.vv\n")
        self.fout.write(f"  #define VXOR       vxor.vv\n")
        self.fout.write(f"  #define BNE        bne\n")
        self.fout.write(f"  #define BEQ        beq\n")
        self.fout.write(f"  #define ADDI       addi\n")
        self.fout.write(f"  #define ADD        add\n")
        self.fout.write(f"  #define SUB        sub\n")
        self.fout.write(f"  #define SD         sd\n")
        self.fout.write(f"  #define LD         ld\n")
        self.fout.write(f"  #define LI         li\n")
        self.fout.write(f"  #define VSETVLI    vsetvli\n")
        self.fout.write(f"  #define DIV        div\n")
        self.fout.write(f"  #define MUL        mul\n")
        self.fout.write(f"  #define FCVT       fcvt.s.w\n")
        self.fout.write(f"  #define FLW        flw\n")
        self.fout.write(f"  #define FEQ        feq.s\n")
        self.fout.write(f"  #define VFMV       vfmv.v.f\n")
        self.fout.write(f"\n")


    def __loadA(self, prfm=False):
        prfm_ext=""
        if prfm: prfm_ext = "_prfm"

        for r in range(0, self.mr // self.vl):
            self.fout.write(f"    LOAD A{r}{prfm_ext}, (A{r}_ptr)\n")
        self.fout.write(f"\n")


    def __loadB(self, prfm=False):
        prfm_ext=""
        if prfm: prfm_ext = "_prfm"

        if (self.gather):
            for r in range(0, self.nr // self.vl):
                self.fout.write(f"    LOAD BTMP{r}{prfm_ext}, (B{r}_ptr)\n")
        else:
            desp=0
            for c in range(0, self.nr):
                self.fout.write(f"    FLW BF{c}, {desp}(B0_ptr)\n")
                desp += self.data_type
            if (self.broadcast):
                for c in range(0, self.nr):
                    self.fout.write(f"    VFMV B{c}{prfm_ext}, BF{c}\n")
        self.fout.write(f"\n")


    def __updateA_address(self):
        for r in range(0, self.mr // self.vl):
            self.fout.write(f"    ADDI A{r}_ptr, A{r}_ptr, {self.a_dsp}\n")


    def __updateB_address(self):
        for r in range(0, self.nr // self.vl):
            self.fout.write(f"    ADDI B{r}_ptr, B{r}_ptr, {self.b_dsp}\n")
        self.fout.write(f"\n")


    def __loop_kc(self, prfm=False):
        prfm_ext_a = ""
        prfm_ext_b = ""
        if (prfm): prfm_ext_a = "_prfm"
        if (prfm) and (self.broadcast): prfm_ext_b = "_prfm"
        
        if self.edge: prfm=False
        load_prfm=prfm
        if (self.pipelining) and (not self.edge): load_prfm = not prfm
    
        #Load A and B    
        if self.reorder:
            self.__loadB(load_prfm)
            self.__loadA(load_prfm)
        else:
            self.__loadA(load_prfm)
            self.__loadB(load_prfm)

        #Gahter for B
        if (self.gather):
            prfm_gather_ext=""
            if (prfm): prfm_gather_ext="_prfm"
            for c in range(0, self.nr):
                self.fout.write(f"    GATHER B{c}, BTMP{c // self.vl}{prfm_gather_ext}, {c % self.vl}\n")
            self.fout.write(f"\n")

        #Update A & B addresses
        self.__updateA_address()
        self.__updateB_address()

        b_ext=""
        if (not self.broadcast) and (not self.gather): 
            b_ext="F"
        
        #macc loop
        for r in range(0, self.mr // self.vl):
            for c in range(0, self.nr):
                    self.fout.write(f"    VFMACC C{r}{c}, B{b_ext}{c}{prfm_ext_b}, A{r}{prfm_ext_a}\n")
        self.fout.write(f"\n")


    def __macro_loop_KC(self):
        self.fout.write(f"  .macro KERNEL_{self.mr}x{self.nr}\n")

        self.__loop_kc()
    
        self.fout.write(f"  .endm\n")
        self.fout.write(f"\n")


    def __loop_pipelining_KC(self):
        self.fout.write(f"  PREV_LOOP_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    BEQ kc_iter, zero, LOOP_LEFT_{self.mr}x{self.nr}\n")
        self.fout.write(f"  LOOP_{self.mr}x{self.nr}:\n")
 
        #Iteration x1
        self.__loop_kc(prfm=True)

        #Iteration x2
        self.__loop_kc(prfm=False)
 
        self.fout.write(f"    ADDI kc_iter, kc_iter, -1\n")
        self.fout.write(f"    BNE kc_iter, zero, LOOP_{self.mr}x{self.nr}\n")
        self.fout.write(f"\n")
  
        self.fout.write(f"    BEQ kc_left, zero, STORE_{self.mr}x{self.nr}\n")
        self.fout.write(f"  LOOP_LEFT_{self.mr}x{self.nr}:\n")

        if (self.gather):
            for c in range(0, self.nr):
                self.fout.write(f"    GATHER B{c}, BTMP{c // self.vl}_prfm, {c % self.vl}\n")
            self.fout.write(f"\n")

        b_ext=""
        prfm_ext_b = ""
        prfm_ext_a = "_prfm"
        if (not self.broadcast) and (not self.gather): 
            b_ext="F"
            self.__loadB()

        if (self.broadcast): prfm_ext_b = "_prfm"

        for r in range(0, self.mr // self.vl):
            for c in range(0, self.nr):
                self.fout.write(f"    VFMACC C{r}{c}, B{b_ext}{c}{prfm_ext_b}, A{r}{prfm_ext_a}\n")
        self.fout.write(f"\n")


    def __start_function_stack(self):
        self.fout.write(f"  gemm_ukernel_asm_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    //Stack save registers\n")
        self.fout.write(f"    ADDI sp, sp, -56\n")
        self.fout.write(f"    SD s6,  48(sp)\n")
        self.fout.write(f"    SD s5,  40(sp)\n")
        self.fout.write(f"    SD s4,  32(sp)\n")
        self.fout.write(f"    SD s3,  24(sp)\n")
        self.fout.write(f"    SD s2,  16(sp)\n")
        self.fout.write(f"    SD s1,   8(sp)\n")
        self.fout.write(f"    SD s0,   0(sp)\n")
        self.fout.write(f"\n")
    
    
    def __end_function_stack(self):
        self.fout.write(f"  END:\n")
        self.fout.write(f"    //Stack restore registers\n");
        self.fout.write(f"    LD s6, 48(sp)\n");
        self.fout.write(f"    LD s5, 40(sp)\n");
        self.fout.write(f"    LD s4, 32(sp)\n");
        self.fout.write(f"    LD s3, 24(sp)\n");
        self.fout.write(f"    LD s2, 16(sp)\n");
        self.fout.write(f"    LD s1,  8(sp)\n");
        self.fout.write(f"    LD s0,  0(sp)\n");
        self.fout.write(f"    ADDI sp, sp, 56\n");
        self.fout.write(f"    ret\n");
    
    
    def __set_vl(self):
        self.fout.write(f"    LI t1, {self.vl}\n")
        self.fout.write(f"    VSETVLI {self.regs[1]}, {self.regs[1]}, e32, m1\n")
        self.fout.write(f"\n")
    
    
    def __C_address(self):
        MR = self.mr
        NR = self.nr
        #C Address ldC strides
        for c in range(1, NR):
            self.fout.write(f"    ADD C0{c}_ptr, C0{c-1}_ptr, ldC\n")
        self.fout.write(f"\n")
    
   
    def __AB_address(self):
        MR_row = self.mr // self.vl
        NR_row = self.nr // self.vl

        for c in range(1, self.mr // self.vl):
            self.fout.write(f"    ADDI A{c}_ptr, A{c-1}_ptr, {self.vl * self.data_type}\n")
        self.fout.write(f"\n")

        #B Address stride
        for b in range(1, NR_row):
            self.fout.write(f"    ADDI B{b}_ptr, B{b-1}_ptr, {self.vl * self.data_type}\n")
        self.fout.write(f"\n")

    def __load_C_to_vregs(self):
        MR = self.mr
        NR = self.nr

        MR_rows = MR // self.vl
        NR_rows = NR // self.vl
          
        #---------------------------------------- 
        #Software pipelining and unroll controll
        #---------------------------------------- 
        if (not self.edge) and (self.pipelining):
            if self.reorder:
                if (self.gather) or (self.broadcast): self.__loadB(prfm=True)
                self.__loadA(prfm=True)
            else:
                self.__loadA(prfm=True)
                if (self.gather) or (self.broadcast): self.__loadB(prfm=True)

        if self.unroll != 0:
            self.fout.write(f"    LI unroll, {self.unroll}\n")
            self.fout.write(f"    DIV kc_iter, kc, unroll\n")
            self.fout.write(f"    MUL kc_left, kc_iter, unroll\n")
            self.fout.write(f"    SUB kc_left, kc, kc_left\n")
            self.fout.write(f"\n")

        if (not self.edge) and (self.pipelining):
            self.__updateA_address()
            if (self.gather) or (self.broadcast): self.__updateB_address()
        #---------------------------------------- 
             
        self.fout.write(f"    FCVT  ft8, zero\n")
        self.fout.write(f"    FLW   BETA, (beta_ptr)\n")
        self.fout.write(f"    FEQ   TMP, BETA, ft8\n")
        self.fout.write(f"    BEQ   TMP, zero, LOAD_{self.mr}x{self.nr}\n")
        self.fout.write(f"\n")
         
        #BETA=0: Initialize Register to 0
        for c in range(0, NR):
            self.fout.write(f"    VXOR C0{c}, C0{c}, C0{c}\n")
            desp = 0
            for r in range(1, MR_rows):
                self.fout.write(f"    VXOR C{r}{c}, C{r}{c}, C{r}{c}\n")
                desp += self.vl * self.data_type
            if desp > 0:
                self.fout.write(f"    ADDI C0{c}_ptr, C0{c}_ptr, {desp}\n")
            self.fout.write(f"\n")
        self.fout.write(f"\n")
        self.fout.write(f"    j PREV_LOOP_{MR}x{NR}\n")
        self.fout.write(f"\n")
    
        #BETA=1: Store C to registers
        self.fout.write(f"  LOAD_{MR}x{NR}:\n")
        for c in range(0, NR):
            self.fout.write(f"    LOAD C0{c}, (C0{c}_ptr)\n")
            desp = self.vl * self.data_type
            for r in range(1, MR_rows):
                self.fout.write(f"    ADDI C0{c}_ptr, C0{c}_ptr, {desp}\n")
                self.fout.write(f"    LOAD C{r}{c}, (C0{c}_ptr)\n")
                desp = self.vl * self.data_type
            self.fout.write(f"\n")
        self.fout.write(f"\n")
    
   
    def __simple_loop_KC(self):
        self.fout.write(f"  PREV_LOOP_{self.mr}x{self.nr}:\n")
        self.fout.write(f"  LOOP_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    KERNEL_{self.mr}x{self.nr}\n")
        self.fout.write(f"    ADDI kc, kc, -1\n")
        self.fout.write(f"    BNE kc, zero, LOOP_{self.mr}x{self.nr}\n")
        self.fout.write(f"\n")


    def __unroll_loop_KC(self):
        self.fout.write(f"  PREV_LOOP_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    BEQ kc_iter, zero, LOOP_LEFT_{self.mr}x{self.nr}\n")
        self.fout.write(f"  LOOP_{self.mr}x{self.nr}:\n")
        for i in range(0,self.unroll):
            self.fout.write(f"    KERNEL_{self.mr}x{self.nr}\n")
        self.fout.write(f"    ADDI kc_iter, kc_iter, -1\n")
        self.fout.write(f"    BNE  kc_iter, zero, LOOP_{self.mr}x{self.nr}\n")
        self.fout.write(f"\n")
        self.fout.write(f"    BEQ kc_left, zero, STORE_{self.mr}x{self.nr}\n")
        self.fout.write(f"  LOOP_LEFT_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    KERNEL_{self.mr}x{self.nr}\n")
        self.fout.write(f"    ADDI kc_left, kc_left, -1\n")
        self.fout.write(f"    BNE kc_left, zero, LOOP_LEFT_{self.mr}x{self.nr}\n")
        self.fout.write(f"\n")
        
    
    def __store_C_from_vregs(self):
        self.fout.write(f"  STORE_{self.mr}x{self.nr}:\n")
        for c in range(0, self.nr):
            self.fout.write(f"    STORE C{self.mr // self.vl - 1}{c}, (C0{c}_ptr)\n")
            desp = self.vl * self.data_type
            for r in range(self.mr // self.vl - 1, 0, -1):
                self.fout.write(f"    ADDI C0{c}_ptr, C0{c}_ptr, -{desp}\n")
                self.fout.write(f"    STORE C{r-1}{c}, (C0{c}_ptr)\n")
                desp = self.vl * self.data_type
            self.fout.write(f"\n")
        self.fout.write(f"\n")


    def generate_umicro(self):
        self.__set_output()
        self.__header()
        self.__registers_to_macros()

        if (self.edge) or (not self.pipelining):
            self.__macro_loop_KC()

        self.__start_function_stack()
        self.__set_vl()
        self.__AB_address()
        self.__C_address()
        self.__load_C_to_vregs()

        if (not self.edge) and (self.pipelining):
            self.__loop_pipelining_KC()
        else:
            if (self.unroll == 0):
                self.__simple_loop_KC()
            else:
                self.__unroll_loop_KC()

        self.__store_C_from_vregs()
        self.__end_function_stack()
    
#---------------------------------------------------------------------
#---------------------------------------------------------------------

