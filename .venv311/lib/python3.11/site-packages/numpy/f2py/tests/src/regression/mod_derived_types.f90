module mtypes
  implicit none
  integer, parameter :: value1 = 100
  type :: master_data
    integer :: idat = 200
  end type master_data
  type(master_data) :: masterdata
end module mtypes


subroutine no_type_subroutine(ain, aout)
  use mtypes, only: value1
  integer, intent(in) :: ain
  integer, intent(out) :: aout
  aout = ain + value1
end subroutine no_type_subroutine

subroutine type_subroutine(ain, aout)
  use mtypes, only: masterdata
  integer, intent(in) :: ain
  integer, intent(out) :: aout
  aout = ain + masterdata%idat
end subroutine type_subroutine