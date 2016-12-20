subroutine test_sub(k)
    implicit none

  !===============================
  !   This is a test subroutine
  !===============================

    integer, intent(in)           :: k
    double precision, allocatable :: a(:)
    integer, parameter            :: nmax=10
    integer                       :: i

    allocate (a(nmax))

    do i=1,nmax
      a(i) = dble(i)*5.d0
    enddo

    print *, 'Hello world'
    write (*,*) a(:)

end subroutine test_sub
