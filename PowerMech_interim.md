<a name="br1"></a> 

Power Mechanism

ponkshekaustubh11

May 2023

1 Problem Statement

Given a dataset with n samples X

∈ R<sup>k</sup>, a positive integer p ∈ Z+, and an operator

k×n

H

: X

→ Z

such that Z = H(X )<sup>p</sup>X ; what are the required conditions that

k×1

k×k

k×1

i

i

need to be satisﬁed by H(X) and p to formally guarantee that Z is ϵ-Lipschitz private

with respect to the dataset X ?

To avoid confusion, we restate that H(X) denotes a matrix whose entries depend on

X. In the rest of the paper we use H(X) and H interchangeably to mean the same thing

without any loss of generality.

[Decorrelating privacy theorem] For X ∈ R<sup>k</sup> distributed as ∼ f (x), applying Z =

X

p

H(X)<sup>p</sup>X guarantees ϵ-Lipschitz privacy through Z when the integer power p satisﬁes

~~°~~

~~°~~

~~°~~

~~°~~

~~°~~ <sub>f</sub> ′<sub>(X)</sub> °

° <sub>f</sub> ′(X)

~~°~~

ϵ−~~°~~

°

°

−ϵ~~°~~

f (X)

f (X)

~~°~~

~~°~~ ≥ p ≥ ~~°~~

~~°~~

~~°~~

~~°~~

°

°

°

°

° <sub>f ′(X)</sub> °

H(X)−1 ∂

H(X)

H(X)−

1 ∂H(X)

∂X

(2

−1)

°

°

°

°

°

°

∂X

f (X)

Proof. The equation Z = [H(X )]<sup>p</sup>X can be unrolled as

p<sub>i</sub>

i

Z<sub>pi</sub> = g<sub>p</sub> ◦ g<sub>p−1</sub> ◦···◦ g (X )

(1)

1

i

where g ◦ g

(·) = H(X).g

(·).

p−1

p

p−1

If g is a one-to-one function on the support of X whose pdf is given by f (x) where

X

x ∈ R<sup>k</sup>, then the pdf of Z = g(X) is

h(Z) = f (g− (Z))|det(J(g<sup>−1</sup>(Z)))|

1

X

for Z in the range of g, where J(X) is the Jacobian matrix of g that is evaluated at

X. This is classically known as the multidimensional change of variable theorem in

the context of probability density functions. But since we have g ◦ g

◦ ··· ◦ g<sub>1</sub>(X)

p−1

p

instead of a single g(·), this can be written as

¯

¯

¯

dg<sub>−</sub><sup>1</sup> ¯

¯

<sup>p</sup> ¯

h (Z ) = h

(g− (Z ))¯det

1

¯

p

p

p−1

p

p

¯

dZ

¯

p

We can rearrange the Jacobian of our iteration as follows

<sub>∂H</sub><sup>p</sup>X

∂Z<sub>p−1</sub>

∂<sup>Hp X</sup>

<sub>∂</sub>H<sup>p</sup>−<sup>1</sup> X

∂<sup>Zp</sup>

∂Z<sup>p</sup>−1

∂<sup>HHp−1X</sup>

\=

\=

\=

= J(Z<sub>(p−1)i</sub> )

∂H<sup>p</sup>−<sup>1</sup>X

1



<a name="br2"></a> 

Let us ﬁnd this jacobian matrix J. For that consider the equation

<sup>Z</sup>p <sup>= H(Z</sup><sub>p−1</sub>)Z<sub>p−1</sub>

X<sup>k</sup>

∴ Z<sub>pi</sub>

Since the Jacobian matrix J is

\=

H(Z

) Z

p−1 i j <sup>p−1</sup>j

j=1

<sup>∂Zp</sup>i

∂Z<sub>p−1j</sub>

J<sub>i j</sub>

\=

~~P~~

k

H Z<sub>p</sub> 1<sup>)</sup>

(

∂

Z<sub>p−1l</sub>

l~~=~~1

−

il

J<sub>i j</sub>

\=

∂Z<sub>p−1j</sub>

X<sup>k</sup> <sub>∂</sub>H(Z<sub>p−1</sub>)<sub>il</sub>

<sup>J</sup>i j

\=

Z

p−1<sub>l</sub>

\+ H(Z

p−1 i j

)

<sup>∂Zp</sup>−<sup>1</sup><sub>j</sub>

l=1

¯

¯

¯

¯

³

´

<sup>−1</sup>. Therefore upon apply-

¯

dg<sub>p</sub>

dZ

<sub>1</sub>¯

¯

dg<sub>p</sub>

¯

−

But we have the following: ¯det

¯ = ¯det

¯

¯

¯

¯

¯

dz

p−1

p−1

ing log to the result of the change of variable theorem in our case, we get

¯

¯

¯

¯

¯

dg

dZ

¯

¯

dg<sub>p</sub>

¯

p−1

p−2

<sup>= logh</sup>p−2<sup>(Z</sup><sub>p−2</sub>)−log

¯det

¯

log¯det

¯

−

¯

¯

¯

¯

<sup>dZ</sup>p−1

= ...

¯

¯

X<sup>p</sup>

¯

dg<sub>i</sub>

¯

= logh (Z )− log

¯det

¯

0

0

¯

¯

<sup>dZ</sup>i−1

i=1

Therefore we have that the logarithm of the ratio of the probability densities before and

after P iterations as

µ

¶

h(Z)

f (X)

X<sup>p</sup>

Y<sup>p</sup>

log

= −

log|detJ(Z<sub>(p−1)i</sub> | = −log( |detJ(Z<sub>(p−1)i</sub> |)

p=1

i=1

Now applying the derivative to the log probability and taking its norm and setting it

to be less than ϵ we get the following required condition in order to satisfy Lipschitz

privacy

°

°

°

°

°

∂|

<sup>detJ(Z</sup>(p

|

°

−

\1)

i

X<sup>p</sup>

°

°

° f ′(X)

°

∂

∂X

<sup>X</sup>i

°

logh(Z)°

°

∂

°

\=

−

°

°

°

°

° f (X)

|detJ(Z<sub>(p−1)</sub> |°

p=1

i

Now to differentiate the determinant of a matrix, we use Jacobi’s formula

°

°

°

°

°

p

°

°

°

f ′(X)

X

∂

°

log( det(J))<sup>°</sup>

°

logh(Z)°

= °

−

|

°

°

°

∂X

° f (X)

°

p=1

°

°

∂J

<sup>(Z</sup>(p

\1)

i

)

°

°

°

°

X<sup>p</sup>

det(J(Z

)tr(J−1

−

)

°

°

° f ′(X)

(p 1)<sub>i</sub>

°

∂

∂X

−

<sup>X</sup>i

∂

°

logh(Z)°

°

°

\=

−

°

°

°

°

° f (X)

|detJ(Z<sub>(p−1)i</sub> |

°

p=1

2



<a name="br3"></a> 

∂

J(Z

∂X<sub>i</sub>

(p−1)

i

)

Finally, we need to evaluate the term J<sup>′</sup> =

∂(<sup>P</sup>

(

<sub>1</sub>)<sub>lm</sub><sup>)</sup>

∂

H(Z

Z

p−1

<sup>)</sup>ln

k

<sup>Zp</sup>−<sub>1</sub> + H Z<sub>p</sub>

∂X<sub>i</sub>

−

∂J(Z<sub>(p−1)</sub> )

n

′

lm

n=1

∂

p−1m

J

\=

i

\=

lm

∂X<sub>i</sub>

X<sup>k</sup>

2

H Z

(

1<sup>)</sup>

H Z

∂Z<sub>p 1m</sub>

(

1<sup>)</sup>

<sub>∂</sub>Z<sub>p</sub>

ln

′

p

−

−

ln

∂

p

−

−

<sup>1</sup>n

J

\=

(<sup>∂</sup>

Z<sub>p−1n</sub>

\+

)

lm

∂X<sub>i</sub>∂Z<sub>p 1m</sub>

∂X<sub>i</sub>

n=1

−

Therefore for obtaining ϵ−Lipschitz privacy, we need to have

°

°

°

°

°

p

°

°

°

f ′(X)

X

∂

∂X

°

°

°

logh(Z)°

log(<sub>|</sub>det(J))

= °

−

−

° ≤ ϵ

°

°

° f (X)

°

p=1

°

°

°

°

°

°

°

°

°

°

° f ′(X)

J° ° f ′(X)°

°

J°

∂

∂X

∂

∂X

∂

°

logh(Z)°

°

pJ−1

°

°

°

°pH−1

°

\=

≤

\+

≤ ϵ

°

°

°

°

°

°

°

°

f (X)

f (X)

∂X

2 Estimating sample probability

We use Kernel Density Estimation for estimating the probability density of each sam-

ple.

1

nh<sup>d</sup>

X<sup>n</sup>

<sup>x− X</sup>i

h

ˆ

f (x) =

K(

)

i=1

The Gaussian kernel is given by

e−||u||<sup>2</sup>

(2π)<sup>d/2</sup>

K(u) =

However, we need to ﬁnd conﬁdence intervals for these probability density esti-

mates. The range in which the true probability density lies with 1− α probability is

given by

1−α/2<sup>s</sup>

<sub>1−α/2</sub>s

nh<sup>d</sup>

ˆ

ˆ

µ<sub>K</sub> f (x)

µ<sub>K</sub> f (x)

ˆ

ˆ

CI<sub>1−α</sub> = [f (x)− z

<sup>The term µ</sup><sub>K</sub> is given by

, f (x)+ z

]

nh<sup>d</sup>

µ<sub>K</sub> = <sup>Z</sup>K<sup>2</sup>(x)dx

For Gaussian kernel, this evaluates to

µ<sub>K</sub> = 1/(2<sup>d</sup>π<sup>d/2</sup>)

The conﬁdence bound for the gradient of is given by

s

ˆ

³

´

∂f (x) ∂f (x)

∂x<sub>i</sub>

1

−

= O(h<sup>2</sup>)+O<sub>P</sub>

∂x<sub>i</sub>

nh<sup>d</sup>+2

q

ˆ

ˆ

f (x) = f (x)+ K f (x)N (0,1)

s

ˆ

∂f (x) ∂f (x)

K

ˆ

\=

\+

f (x)N (0,1)

∂x<sub>i</sub>

∂x<sub>i</sub>

4f (x)

ˆ

3



<a name="br4"></a> 

3 Bringing it together

The condition for ϵ Lipschitz Privacy is given by

°

°

°

°

∂

∂X

°

logh(Z)°

≤ ϵ

°

°

For obtaining Lipschitz privacy on estimated probability with 1−α conﬁdence,

°

°

°

°

°

°

°

p

°

° ˆ

ˆ

p

°

°

°

f ′(X)

X

f ′(X) f ′(X)− f ′(X)

X

∂

∂X

°

∂

log( det(J))<sup>°</sup>

°

∂

<sub>∂</sub>X

°

°

logh(Z)°

log(<sub>|</sub>det(J))

= °

−

|

° = °

\+

−

° ≤ ϵ

°

°

° f (X) <sub>∂</sub>X

°

° f (X)

f (X)

°

p=1

p=1

°

°

°

°

°

°

° ˆ

p

°

°

ˆ

°

°

°

f ′(X)

X

f ′(X)− f ′(X)

∂

∂X

°

∂

°

°

°

°

logh(Z)°

<sup>log(</sup>|<sup>det(J))</sup>

≤ °

−

°+°

° ≤ ϵ

°

°

° f (X) <sub>∂</sub>X

°

°

f (X)

°

p=1

Now let’s use the conﬁdence interval founds on f (X) to estimate ϵ

°

° °

°

°

° °

°

°

° °

°

µ

¶

°

ˆ

p

° °

ˆ

p

°

f ′(X)

X

f ′(X)

X

°

° °

log(|det(J))<sup>°</sup>

Let ϵ<sup>′</sup> = max

−

log(|det(J)) ,

−

°

r

° °

r

°

°

° °

°

ˆ

p

1

ˆ

p

\=

1

° ˆ

f (x)

\=

° ° ˆ

f (x)

°

<sup>f (x)− z</sup>1−α/2

µK

nh<sup>d</sup>

° °

<sup>f (x)+ z</sup>1−α/2

µK

nh<sup>d</sup>

°

°

°

°

°

ˆ

°

f ′(X)− f ′(X)

<sub>∴ ϵ = ϵ′ +</sub>°

°

°

°

°

f (X)

°

4

