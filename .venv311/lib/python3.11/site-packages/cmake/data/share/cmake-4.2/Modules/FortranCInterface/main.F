        program main
          call call_sub()
#ifdef CALL_MOD
          call call_mod()
#endif
        end
