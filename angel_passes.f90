module angel_passes
  use iso_c_binding
  implicit none
contains
  
  subroutine update(alpha, beta, mat, sounds, n) bind(c)
    
    Integer :: i 
    integer(c_int) :: n
    real(c_double) :: s, p
    real(c_double) :: alpha, beta
    real(c_double) :: mat(n, n) 
    real(c_double) :: sounds(n) 
    real(c_double) :: sounds_(n)
    real(c_double) :: random(n) 
  
    call random_number(random(:))
    
    do i=1, n
       s = sum(mat(:, i)*sounds(:))
       p = (1.0d0-(alpha+beta))*s + alpha
       if(random(i)<p) then
          sounds_(i) = 1.0
       else
          sounds_(i) = 0.0
       end if
    end do
  
    sounds(:) = sounds_(:)
  
  end subroutine update
  
  subroutine batch_update(alpha, beta, mat, sounds, n, steps) bind(c)
    
    integer :: step 
    integer(c_int) :: steps, n
    real(c_double) :: alpha, beta
    real(c_double) :: mat(n, n) 
    real(c_double) :: sounds(n) 
  
    do step=1, steps
       call update(alpha, beta, mat, sounds, n)
    end do
  
  end subroutine batch_update

end module angel_passes
