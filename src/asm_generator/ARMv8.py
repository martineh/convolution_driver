import common as cm

#=====================================================================
#                           MODULE ARCH #1                           |
#=====================================================================
#|          Architecture Configuration          |       ARMv8        |
#=====================================================================
class ASM_ARMv8():

    def __init__(self, MR, NR, arch, unroll, pipelining, tag=""):
        self.tag        = tag
        self.edge       = False
        self.vl         = 4  
        self.vr_max     = 32 
        self.data_type  = 4
        self.arch       = arch
        self.pipelining = pipelining

        if pipelining:
            self.unroll = 2
        else:
            self.unroll = unroll

        self.maxB       = NR // self.vl
         
        self.orig_mr    = MR
        self.orig_nr    = NR
        self.mr         = MR
        self.nr         = NR

        self.r_kc       = "x0"
        self.r_alpha    = "x1"
        self.r_a        = "x2"
        self.r_b        = "x3"
        self.r_beta     = "x4"
        self.r_c        = "x5"
        self.r_ldC      = "x6"
        self.r_unroll   = "x7"
        self.r_kc_iter  = "x8"
        self.r_kc_left  = "x9"

        self.AB_vreg    = 0

        self.fout = None

    def set_edge(self, new_MR, new_NR):
        if  (new_MR != 0) and (new_NR != 0) and (new_MR % self.vl == 0) and (new_NR % self.vl == 0):
            self.edge = True
            self.mr   = new_MR
            self.nr   = new_NR
            return 1
        else:
            return -1

    def set_output(self):
        cm.add_header(self.mr, self.nr)
        self.fout = cm.new_micro_file(self.mr, self.nr)

    def header(self):
        self.fout.write(f"  .text\n")
        self.fout.write(f"  .align      5\n")
        self.fout.write(f"  .global     gemm_ukernel_asm_{self.mr}x{self.nr}\n")
        self.fout.write(f"  //void gemm_ukernel_asm(size_t kc, float *alpha, float *a, float *b, float *beta, float *c, size_t ldC); \n")
        self.fout.write(f"    // register arguments:\n")
        self.fout.write(f"    // x0   kc\n")
        self.fout.write(f"    // x1   alpha\n")
        self.fout.write(f"    // x2   a\n")
        self.fout.write(f"    // x3   b\n")
        self.fout.write(f"    // x4   beta\n")
        self.fout.write(f"    // x5   c\n")
        self.fout.write(f"    // x6   ldC\n")

    def registers_to_macros(self):
        MR = self.mr
        NR = self.nr

        self.fout.write(f"  #define kc         {self.r_kc}\n")
        self.fout.write(f"  #define alpha_ptr  {self.r_alpha}\n")
        self.fout.write(f"  #define A_ptr      {self.r_a}\n")
        self.fout.write(f"  #define B_ptr      {self.r_b}\n")
        self.fout.write(f"  #define beta_ptr   {self.r_beta}\n")
        self.fout.write(f"  #define C00_ptr    {self.r_c}\n")
        self.fout.write(f"  #define ldC        {self.r_ldC}\n")
        self.fout.write(f"  #define unroll     {self.r_unroll}\n")
        self.fout.write(f"  #define kc_iter    {self.r_kc_iter}\n")
        self.fout.write(f"  #define kc_left    {self.r_kc_left}\n")
        self.fout.write(f"\n")

        reg    = 10
        vreg   = 0
        tmpreg = 0
        #C with ldC strides Address storage 
        for c in range(1, NR):
            self.fout.write(f"  #define C0{c}_ptr    x{reg}\n")
            reg += 1
        self.fout.write(f"\n")

        #C to vector registers
        MR_rows = MR // self.vl
        for r in range(0, MR_rows):
            for c in range(0, NR):
                self.fout.write(f"  #define C{r}{c}q       q{vreg}\n")
                self.fout.write(f"  #define C{r}{c}v       v{vreg}.4s\n")
                vreg += 1
        self.fout.write(f"\n")
   
        self.AB_vreg = vreg
        #A to vector registers
        for r in range(0, MR_rows):
            self.fout.write(f"  #define A{r}q        q{vreg}\n")
            self.fout.write(f"  #define A{r}v        v{vreg}.4s\n")
            vreg += 1
        self.fout.write(f"\n")

        #B to vector registers
        for r in range(0, self.maxB):
            self.fout.write(f"  #define B{r}q        q{vreg}\n")
            self.fout.write(f"  #define B{r}v        v{vreg}.s\n")
            vreg += 1
        self.fout.write(f"\n")

        if self.pipelining and (not self.edge):
            MR_rows = MR // self.vl

            #A to vector registers
            for r in range(0, MR_rows):
                self.fout.write(f"  #define A_prfm{r}q        q{vreg}\n")
                self.fout.write(f"  #define A_prfm{r}v        v{vreg}.4s\n")
                vreg += 1
            self.fout.write(f"\n")

            #B to vector registers
            for r in range(0, self.maxB):
                self.fout.write(f"  #define B_prfm{r}q        q{vreg}\n")
                self.fout.write(f"  #define B_prfm{r}v        v{vreg}.s\n")
                vreg += 1
            self.fout.write(f"\n")
              
            
        self.fout.write(f"  #define TMP        s11\n")
        self.fout.write(f"  #define BETA       s10\n")
        self.fout.write(f"\n")

    def macro_loop_KC(self):
        MR = self.mr
        NR = self.nr

        #KC Loop (MAC)
        MR_rows = MR // self.vl
        NR_rows = NR // self.vl

        self.fout.write(f"  .macro KERNEL_{MR}x{NR}{self.tag}\n")
       
        #1-Load A
        desp = self.vl * self.data_type
        for r in range(0, MR_rows):
            self.fout.write(f"    ldr A{r}q,  [A_ptr]\n")
            self.fout.write(f"    add A_ptr, A_ptr, #{desp}\n")
        self.fout.write(f"\n")

        #2-Load B
        desp = self.vl * self.data_type
        for c in range(0, NR_rows):
            self.fout.write(f"    ldr B{c}q,  [B_ptr]\n")
            self.fout.write(f"    add B_ptr, B_ptr, #{desp}\n")
        self.fout.write(f"\n")
        
        #3-MAC-LOOP (B Loads + MAC)
        for r in range(0, MR_rows):
            for c in range(0, NR_rows):
                for v in range(0, self.vl):
                    self.fout.write(f"    fmla C{r}{(c*self.vl) + v}v, A{r}v, B{c}v[{v}]\n")
   
        #4-Edge Call? Correct pointer position for A and B.
        if (self.edge):
            inc = (self.orig_mr - self.mr) * self.data_type
            if inc > 0:
                self.fout.write(f"    add A_ptr, A_ptr, #{inc}\n")

            inc = (self.orig_nr - self.nr) * self.data_type
            if inc > 0:
                self.fout.write(f"    add B_ptr, B_ptr, #{inc}\n")
        
        self.fout.write(f"  .endm\n")
        self.fout.write(f"\n")

    def start_function_stack(self):
        self.fout.write(f"  gemm_ukernel_asm_{self.mr}x{self.nr}:\n")
        self.fout.write(f"    add	sp, sp, #-(11 * 16)\n")
        self.fout.write(f"    stp	d8, d9,   [sp, #(0 * 16)]\n")
        self.fout.write(f"    stp	d10, d11, [sp, #(1 * 16)]\n")
        self.fout.write(f"    stp	d12, d13, [sp, #(2 * 16)]\n")
        self.fout.write(f"    stp	d14, d15, [sp, #(3 * 16)]\n")
        self.fout.write(f"    stp	d16, d17, [sp, #(4 * 16)]\n")
        self.fout.write(f"    stp	x18, x19, [sp, #(5 * 16)]\n")
        self.fout.write(f"    stp	x20, x21, [sp, #(6 * 16)]\n")
        self.fout.write(f"    stp	x22, x23, [sp, #(7 * 16)]\n")
        self.fout.write(f"    stp	x24, x25, [sp, #(8 * 16)]\n")
        self.fout.write(f"    stp	x26, x27, [sp, #(9 * 16)]\n")
        self.fout.write(f"    str	x28,      [sp, #(10 * 16)]\n")
        self.fout.write(f"\n")


    def end_function_stack(self):
        self.fout.write(f"  END:\n")
        self.fout.write(f"    mov	x0, #0\n")
        self.fout.write(f"    ldp	d8, d9, [sp, #(0 * 16)]\n")
        self.fout.write(f"    ldp	d10, d11, [sp, #(1 * 16)]\n")
        self.fout.write(f"    ldp	d12, d13, [sp, #(2 * 16)]\n")
        self.fout.write(f"    ldp	d14, d15, [sp, #(3 * 16)]\n")
        self.fout.write(f"    ldp	d16, d17, [sp, #(4 * 16)]\n")
        self.fout.write(f"    ldp	x18, x19, [sp, #(5 * 16)]\n")
        self.fout.write(f"    ldp	x20, x21, [sp, #(6 * 16)]\n")
        self.fout.write(f"    ldp	x22, x23, [sp, #(7 * 16)]\n")
        self.fout.write(f"    ldp	x24, x25, [sp, #(8 * 16)]\n")
        self.fout.write(f"    ldp	x26, x27, [sp, #(9 * 16)]\n")
        self.fout.write(f"    ldr	x28,      [sp, #(10 * 16)]\n")
        self.fout.write(f"    add	sp, sp, #(11*16)\n")
        self.fout.write(f"    ret\n")
        self.fout.close() #End Mirco-kernel

    def C_address(self):
        MR = self.mr
        NR = self.nr

        #C Address ldC strides
        for c in range(1, NR):
            self.fout.write(f"    add C0{c}_ptr, C0{c-1}_ptr, ldC\n")
        self.fout.write(f"\n")

    def load_C_to_vregs(self):
        MR = self.mr
        NR = self.nr
        MR_rows = MR // self.vl
        NR_rows = NR // self.vl

        #---------------------------------------- 
        #ONLY FOR Software Pipelining
        if (not self.edge) and self.pipelining:
            self.fout.write(f"    mov unroll, #{self.unroll}\n")
            #Calculating kc_iters and kc_left with unroll x2
            self.fout.write(f"    udiv kc_iter, kc, unroll\n")
            self.fout.write(f"    msub kc_left, kc_iter, unroll, kc\n")
            self.fout.write(f"\n")

            #1-Load A
            desp = self.vl * self.data_type
            for r in range(0, MR_rows):
                self.fout.write(f"    ldr A_prfm{r}q,  [A_ptr]\n")
                self.fout.write(f"    add A_ptr, A_ptr, #{desp}\n")
            self.fout.write(f"\n")
            #2-Load B
            for c in range(0, NR_rows):
                self.fout.write(f"    ldr B_prfm{c}q,  [B_ptr]\n")
                self.fout.write(f"    add B_ptr, B_ptr, #{desp}\n")
            self.fout.write(f"\n")
        elif self.unroll != 0:
            self.fout.write(f"    mov unroll, #{self.unroll}\n")
            #Calculating kc_iters and kc_left with unroll x2
            self.fout.write(f"    udiv kc_iter, kc, unroll\n")
            self.fout.write(f"    msub kc_left, kc_iter, unroll, kc\n")
            self.fout.write(f"\n")
        #---------------------------------------- 
        
        self.fout.write(f"    ldr BETA, [beta_ptr]\n")
        self.fout.write(f"    fmov TMP, 1.0e+0 \n")
        self.fout.write(f"    fcmp BETA, TMP \n")
        self.fout.write(f"    b.eq .LOAD_{MR}x{NR}{self.tag}\n")
        #fout.write("\n")
    
        #BETA=0: Initialize Register to 0
        for c in range(0, NR):
            for r in range(0, MR_rows):
                self.fout.write(f"    movi C{r}{c}v, 0\n")
        self.fout.write(f"\n")
        self.fout.write(f"    b .S_LOOP_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")
    
        #BETA=1: Store C to registers
        self.fout.write(f"  .LOAD_{MR}x{NR}{self.tag}:\n")
        for c in range(0, NR):
            self.fout.write(f"    ldr C0{c}q, [C0{c}_ptr]\n")
            desp = self.vl * self.data_type
            for r in range(1, MR_rows):
                self.fout.write(f"    ldr C{r}{c}q, [C0{c}_ptr, #{desp}]\n")
                desp += self.vl * self.data_type
            self.fout.write(f"\n")
        self.fout.write(f"\n")
    
    def simple_loop_KC(self):
        MR = self.mr
        NR = self.nr
        self.fout.write(f"  .S_LOOP_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"  .LOOP_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"    KERNEL_{MR}x{NR}\n")
        self.fout.write(f"    sub kc, kc, 1\n")
        self.fout.write(f"    cmp kc, 0\n")
        self.fout.write(f"    b.ne .LOOP_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")

    def unroll_loop_KC(self):
        MR = self.mr
        NR = self.nr
        self.fout.write(f"  .S_LOOP_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"    cmp kc_iter, 0\n")
        self.fout.write(f"    b.eq .LOOP_LEFT_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"  .LOOP_{MR}x{NR}{self.tag}:\n")
        for i in range(0,self.unroll):
            self.fout.write(f"    KERNEL_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"    sub kc_iter, kc_iter, 1\n")
        self.fout.write(f"    cmp kc_iter, 0\n")
        self.fout.write(f"    b.ne .LOOP_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")
        
        self.fout.write(f"    cmp kc_left, 0\n")
        self.fout.write(f"    b.eq .STORE_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"  .LOOP_LEFT_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"    KERNEL_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"    sub kc_left, kc_left, 1\n")
        self.fout.write(f"    cmp kc_left, 0\n")
        self.fout.write(f"    b.ne .LOOP_LEFT_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")
    
    def loop_pipelining_KC(self):
        #Loop KC for software pipelining with unroll x2
        MR = self.mr
        NR = self.nr

        #KC Loop (MAC)
        MR_rows = MR // self.vl
        NR_rows = NR // self.vl

        self.fout.write(f"  .S_LOOP_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"    cmp kc_iter, 0\n")
        self.fout.write(f"    b.eq .LOOP_LEFT_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")
        self.fout.write(f"  .LOOP_{MR}x{NR}{self.tag}:\n")
       
        #Load A
        desp = self.vl * self.data_type
        for r in range(0, MR_rows):
            self.fout.write(f"    ldr A{r}q,  [A_ptr]\n")
            self.fout.write(f"    add A_ptr, A_ptr, #{desp}\n")
        self.fout.write(f"\n")

        #Load B
        desp = self.vl * self.data_type
        for c in range(0, NR_rows):
            self.fout.write(f"    ldr B{c}q,  [B_ptr]\n")
            self.fout.write(f"    add B_ptr, B_ptr, #{desp}\n")
        self.fout.write(f"\n")

        #MAC-LOOP (B Loads + MAC) #Iter+1 (with previus loads)
        for r in range(0, MR_rows):
            for c in range(0, NR_rows):
                for v in range(0, self.vl):
                    self.fout.write(f"    fmla C{r}{(c*self.vl) + v}v, A_prfm{r}v, B_prfm{c}v[{v}]\n")
        self.fout.write(f"\n")
        
        #Load A
        desp = self.vl * self.data_type
        for r in range(0, MR_rows):
            self.fout.write(f"    ldr A_prfm{r}q,  [A_ptr]\n")
            self.fout.write(f"    add A_ptr, A_ptr, #{desp}\n")
        self.fout.write(f"\n")
        #Load B
        for c in range(0, NR_rows):
            self.fout.write(f"    ldr B_prfm{c}q,  [B_ptr]\n")
            self.fout.write(f"    add B_ptr, B_ptr, #{desp}\n")
        self.fout.write(f"\n")
        
        #MAC-LOOP (B Loads + MAC) #Iter+2
        for r in range(0, MR_rows):
            for c in range(0, NR_rows):
                for v in range(0, self.vl):
                    self.fout.write(f"    fmla C{r}{(c*self.vl) + v}v, A{r}v, B{c}v[{v}]\n")
        self.fout.write(f"\n")
        
        #5-Branch Loop 
        self.fout.write(f"    sub kc_iter, kc_iter, 1\n")
        self.fout.write(f"    cmp kc_iter, 0\n")
        self.fout.write(f"    b.ne .LOOP_{MR}x{NR}{self.tag}\n")
        self.fout.write(f"\n")
        self.fout.write(f"  .LOOP_LEFT_{MR}x{NR}{self.tag}:\n")
        self.fout.write(f"    cmp kc_left, 0\n")
        self.fout.write(f"    b.eq .STORE_{MR}x{NR}{self.tag}\n")
        
        #MAC-LOOP (B Loads + MAC) #Left
        for r in range(0, MR_rows):
            for c in range(0, NR_rows):
                for v in range(0, self.vl):
                    self.fout.write(f"    fmla C{r}{(c*self.vl) + v}v, A_prfm{r}v, B_prfm{c}v[{v}]\n")
        self.fout.write(f"\n")
    
    def store_C_from_vregs(self):
        MR = self.mr
        NR = self.nr

        #Store C to registers
        MR_rows = MR // self.vl
        self.fout.write(f"  .STORE_{MR}x{NR}{self.tag}:\n")
        for c in range(0, NR):
            self.fout.write(f"    str C0{c}q, [C0{c}_ptr]\n")
            desp = self.vl * self.data_type
            for r in range(1, MR_rows):
                self.fout.write(f"    str C{r}{c}q, [C0{c}_ptr, #{desp}]\n")
                desp += self.vl * self.data_type
            self.fout.write(f"\n")
        self.fout.write(f"\n")

    def generate_umicro(self):
        self.set_output()
        self.header()
        self.registers_to_macros()

        if (self.edge) or (not self.pipelining):
            self.macro_loop_KC()

        self.start_function_stack()
        self.C_address()
        self.load_C_to_vregs()

        if (not self.edge) and (self.pipelining):
            self.loop_pipelining_KC()
        else:
            if (self.unroll == 0):
                self.simple_loop_KC()
            else:
                self.unroll_loop_KC()

        self.store_C_from_vregs()
        self.end_function_stack()

#---------------------------------------------------------------------
#---------------------------------------------------------------------
