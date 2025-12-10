subroutine call_mod
  !DIR$ NOINLINE
  use mymodule
  use my_module
  call mysub()
  call my_sub()
end subroutine call_mod
