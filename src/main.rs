use std::ops;
use std::cmp;


pub trait Numeric: Sized + ops::Add<Self, Output=Self> + ops::Sub<Self, Output=Self> + ops::Mul<Self, Output=Self> + ops::Div<Self, Output = Self> + ops::Neg<Output = Self> + cmp::PartialEq + cmp::PartialOrd + Clone {
    fn abs(self) -> Self;
    fn epsilon() -> Self; // For derivatives and such
    fn precision() -> Self; // When an answer is considered good enough
    fn fromfloat(_: f64) -> Self;
    fn tofloat(_: Self) -> f64;
}

impl Numeric for f64 {
    fn abs(self) -> f64 {
        return self.abs()
    }

    fn epsilon() -> f64 {
        return 1e-10
    }

    fn tofloat(x: f64) -> f64 {
        return x
    }

    fn fromfloat(x: f64) -> f64 {
        return x
    }

    fn precision() -> f64 {
        return 1e-10
    }
}

fn derivative<N: Numeric>(f: &impl Fn(N)->N, x0: N) -> N {
    return (
        f(x0.clone() + N::fromfloat(0.5)*N::epsilon())
        - f(x0 - N::fromfloat(0.5)*N::epsilon()))
            /N::epsilon()
}

fn integral<N: Numeric>(f: &impl Fn(N)->N, x0: N, x1: N, steps: i32) -> N {
    let mut changesign = false;
    let (mut x0, x1) = if x0 > x1 {
        changesign = true;
        (x1, x0)
    } else {
        (x0, x1)
    };

    let eps = (x1.clone() - x0.clone())/N::fromfloat(steps.into());

    let mut accum = N::fromfloat(0.);
    for _ in 0..steps {
        accum = accum + f(x0.clone() + eps.clone()*N::fromfloat(0.5))*eps.clone();
        x0 = x0 + eps.clone();
    }
    if changesign {
        return -accum
    } else {
        return accum
    }
}

fn newtons_method<N: Numeric>(f: &impl Fn(N)->N, mut x0: N, maxiter: i32) -> Option<N> {
    for _ in 0..maxiter {
        let fx0 = f(x0.clone());
        if fx0.clone().abs() < N::precision() {
            return Some(x0)
        }
        x0 = x0.clone() - fx0/derivative(f, x0)
    }
    return None
}

fn factorial(n: usize) -> usize {
    let mut accum = 1;
    for i in 1..=n {
        accum *= i
    }
    return accum
}

fn power<N: Numeric>(x: N, power: usize) -> N {
    let mut accum = N::fromfloat(1.);
    for _ in 0..power {
        accum = accum * x.clone();
    }
    return accum
}

fn solve_ode<N: Numeric, const ORDER: usize>(ode: &impl Fn(N, [N; ORDER]) -> N, t0: N, mut x0: [N; ORDER], maxsolveiters: i32, step_size: N) -> Option<(N, [N; ORDER])> {
    let last_derivative = newtons_method(&|x| {
        let mut x1 = x0.clone();
        x1[ORDER-1] = x;
        return ode(t0.clone(), x1)
    }, x0[ORDER-1].clone(), maxsolveiters)?;
    x0[ORDER-1] = last_derivative;

    // Taylor series, beggining from lowest to highest term
    // We don't do the last one, as it will be solved using Netwon's method anyways
    for current_order in 0..(ORDER-1) {
        for other_order in (current_order+1)..ORDER {
            let n = other_order - current_order;
            let nfactorial = factorial(n);
            x0[current_order] = x0[current_order].clone() + x0[other_order].clone()*power(step_size.clone(), n)/N::fromfloat(nfactorial as f64)
        }
    }
    return Some((t0 + step_size, x0))
}

fn main() {
    println!("{}", derivative(&|x| x*x, 2.0));
    println!("{}", integral(&|x| derivative(&|x| x*x, x), 0.0, 3.0, 10000));
    println!("{}", derivative(&|x| integral(&|x| x*x, 0.0, x, 10000), 3.0));

    println!("{}", derivative(&|x| x*x, 2.0));
    println!("{}", newtons_method(&|x: f64| x.sin() - 1., 0.0, 10000).unwrap()*2.);

    let mut x0 = [1.0, 0.0, -1.0];     // The last one doesn't matter, the first two are enough
                                       // information
    let mut t0 = 0.0;
    let mut max = 0.0;
    for _ in 0..100000 {
        // i.e. x'' + x = 0, x(0) = 1, x'(0) = 0, equation with x = cos(t) as a solution
        (t0, x0) = solve_ode(&|_t: f64, x: [f64;3]| x[2] + x[0] , t0, x0.clone(), 100, 0.00001).unwrap();
        println!("{:?}: {:?}", t0, x0);
        if x0[0] > max {
            max = x0[0]
        }
    }
    println!("{}", max);
}
