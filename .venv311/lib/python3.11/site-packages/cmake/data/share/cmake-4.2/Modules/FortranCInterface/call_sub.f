        subroutine call_sub
          !DIR$ NOINLINE
          call mysub()
          call my_sub()
        end
