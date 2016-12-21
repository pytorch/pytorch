;; Calculation of Hofstadter's male and female sequences as a list of pairs

(define (hofstadter-male-female n)
(letrec ((female (lambda (n)
           (if (= n 0)
           1
           (- n (male (female (- n 1)))))))
     (male (lambda (n)
         (if (= n 0)
             0
             (- n (female (male (- n 1))))))))
  (let loop ((i 0))
    (if (> i n)
    '()
    (cons (cons (female i)
            (male i))
      (loop (+ i 1)))))))

(hofstadter-male-female 8)

(define (find-first func lst)
(call-with-current-continuation
 (lambda (return-immediately)
   (for-each (lambda (x)
       (if (func x)
           (return-immediately x)))
         lst)
   #f)))
