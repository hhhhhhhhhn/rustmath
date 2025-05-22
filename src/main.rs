use std::ops;
use std::cmp;

#[derive(Debug)]
pub enum MathError {
    NotFound
}

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

pub fn derivative<N: Numeric>(f: &impl Fn(N)->N, x0: N) -> N {
    return (
        f(x0.clone() + N::fromfloat(0.5)*N::epsilon())
        - f(x0 - N::fromfloat(0.5)*N::epsilon()))
            /N::epsilon()
}

pub fn integral<N: Numeric>(f: &impl Fn(N)->N, x0: N, x1: N, steps: i32) -> N {
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

pub fn newtons_method<N: Numeric>(f: &impl Fn(N)->N, mut x0: N, maxiter: i32) -> Result<N, MathError> {
    for _ in 0..maxiter {
        let fx0 = f(x0.clone());
        if fx0.clone().abs() < N::precision() {
            return Ok(x0)
        }
        x0 = x0.clone() - fx0/derivative(f, x0)
    }
    return Err(MathError::NotFound)
}

pub fn factorial(n: usize) -> usize {
    let mut accum = 1;
    for i in 1..=n {
        accum *= i
    }
    return accum
}

pub fn power<N: Numeric>(x: N, power: usize) -> N {
    let mut accum = N::fromfloat(1.);
    for _ in 0..power {
        accum = accum * x.clone();
    }
    return accum
}

pub struct ODESolver<'a, N: Numeric, F: Fn(N, [N; ORDER])->N, const ORDER: usize> {
    ode: &'a F,
    // NOTE: In both entries, the last derivative isn't the actual last derivative
    // it's the last t value's last derivative, which is an ok starting point
    // for actually finding the derivative using Newton's Method,
    // which is done when computing the *next* value.
    // The entries could be actually ORDER-1 in length,
    // (as so could be x0), but this prevents copying,
    // and allows users to put an initial guess for finding the last derivative.
    right_entries: Vec<[N; ORDER]>,
    left_entries: Vec<[N; ORDER]>,
    max_solve_iters: i32,
    t0: N,
    step_size: N,
}

impl<'a, N: Numeric, F: Fn(N, [N; ORDER]) -> N, const ORDER: usize> ODESolver<'a, N, F, ORDER> {
    pub fn new(ode: &'a F, t0: N, x0: [N; ORDER], step_size: N, max_solve_iters: i32) -> Self {
        let result = Self{
            ode,
            t0,
            step_size,
            max_solve_iters,
            left_entries: vec![x0.clone()],
            right_entries: vec![x0]
        };
        return result
    }
    pub fn advance_right(&mut self) -> Result<(), MathError> {
        let mut current_x = self.right_entries.last().expect("Entries always non-empty").clone();
        let current_t = self.current_maximum_t();
        let last_derivative = newtons_method(&|last_derivative| {
            let mut changed_x = current_x.clone();
            changed_x[ORDER-1] = last_derivative;
            return (self.ode)(current_t.clone(), changed_x)
        }, current_x[ORDER-1].clone(), self.max_solve_iters)?;
        current_x[ORDER-1] = last_derivative;

        let mut next_x = current_x;
        // Taylor series, beggining from lowest to highest term
        // We don't do the last one, as it will be solved using Netwon's method anyways
        for current_order in 0..(ORDER-1) {
            for other_order in (current_order+1)..ORDER {
                let n = other_order - current_order;
                let nfactorial = factorial(n);
                next_x[current_order] = next_x[current_order].clone() + next_x[other_order].clone()*power(self.step_size.clone(), n)/N::fromfloat(nfactorial as f64)
            }
        }
        self.right_entries.push(next_x.clone());
        return Ok(())
    }
    pub fn advance_left(&mut self) -> Result<(), MathError> {
        let mut current_x = self.left_entries.last().expect("Entries always non-empty").clone();
        let current_t = self.current_mininum_t();
        let last_derivative = newtons_method(&|last_derivative| {
            let mut changed_x = current_x.clone();
            changed_x[ORDER-1] = last_derivative;
            return (self.ode)(current_t.clone(), changed_x)
        }, current_x[ORDER-1].clone(), self.max_solve_iters)?;
        current_x[ORDER-1] = last_derivative;

        let mut next_x = current_x;
        // Taylor series, beggining from lowest to highest term
        // We don't do the last one, as it will be solved using Netwon's method anyways
        for current_order in 0..(ORDER-1) {
            for other_order in (current_order+1)..ORDER {
                let n = other_order - current_order;
                let nfactorial = factorial(n);
                next_x[current_order] = next_x[current_order].clone() + next_x[other_order].clone()*power(-self.step_size.clone(), n)/N::fromfloat(nfactorial as f64)
            }
        }
        self.left_entries.push(next_x.clone());
        return Ok(())
    }

    fn current_maximum_t(&self) -> N {
        return self.t0.clone() + self.step_size.clone()*N::fromfloat((self.right_entries.len() - 1) as f64);
    }

    fn current_mininum_t(&self) -> N {
        return self.t0.clone() - self.step_size.clone()*N::fromfloat((self.left_entries.len() - 1) as f64);
    }

    pub fn evaluate(&mut self, t: N) -> Result<N, MathError> {
        while t < self.current_mininum_t() {
            self.advance_left()?;
        }
        while t > self.current_maximum_t() {
            self.advance_right()?;
        }
        // TODO: Interpolate between two sorrounding values
        if t > self.t0 {
            let distance = t - self.t0.clone();
            let mut index = N::tofloat(distance.clone()/self.step_size.clone()) as usize;
            println!("Distance {}, index {}", N::tofloat(distance), index);
            if index >= self.right_entries.len() {
                index = self.right_entries.len()-1
            }
            return Ok(self.right_entries[index][0].clone())
        } else {
            let distance = self.t0.clone() - t;
            let mut index = N::tofloat(distance/self.step_size.clone()) as usize;
            if index >= self.left_entries.len() {
                index = self.left_entries.len()-1
            }
            return Ok(self.left_entries[index][0].clone())
        }
    }
}

fn main() {
    println!("{}", derivative(&|x| x*x, 2.0));
    println!("{}", integral(&|x| derivative(&|x| x*x, x), 0.0, 3.0, 10000));
    println!("{}", derivative(&|x| integral(&|x| x*x, 0.0, x, 10000), 3.0));

    println!("{}", derivative(&|x| x*x, 2.0));
    println!("{}", newtons_method(&|x: f64| x.sin() - 1., 0.0, 10000).unwrap()*2.);

    let x0 = [1.0, 0.0, -1.0];     // The last one doesn't matter, the first two are enough
                                       // information
    let t0 = 0.0;
    let ode = |_t: f64, x: [f64;3]| x[2] + x[0];
    let mut solver = ODESolver::new(&ode, t0, x0, 0.00001, 100);
    println!("{:?} {}", solver.evaluate(-2.).unwrap(), (-2.0_f64).cos());
    println!("{:?} {}", solver.evaluate(-1.).unwrap(), (-1.0_f64).cos());
    println!("{:?} {}", solver.evaluate( 0.).unwrap(), ( 0.0_f64).cos());
    println!("{:?} {}", solver.evaluate( 1.).unwrap(), ( 1.0_f64).cos());
    println!("{:?} {}", solver.evaluate( 2.).unwrap(), ( 2.0_f64).cos());
    println!("{:?} {}", solver.evaluate( 1.).unwrap(), ( 1.0_f64).cos());
    println!("{:?} {}", solver.evaluate(99.).unwrap(), (99.0_f64).cos());
}
