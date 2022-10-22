"""Low-level HELAS functions."""

import numpy as np

# double complex fi[5],chi[1]
# double precision p(0:3),sf[1],sfomeg[1],omega[1],fmass,pp,pp3,sqp0p3,sqm(0:1)
# integer nhel,nsf,ip,im,nh

# double precision 0.0, 0.5, 2.0
# parameter( 0.0 = 0.0, 0.5 = 0.5, 2.0 = 2.0 )
def ixxxxx(p, fmass, nhel, nsf, fi):
    fi[4] = (p[0] + 1j * p[3]) * nsf
    fi[5] = (p[1] + 1j * p[2]) * nsf

    sqm = np.zeros((2,), dtype=np.float64)
    sf = np.zeros((2,), dtype=np.float64)
    sfomeg = np.zeros((2,), dtype=np.float64)
    omega = np.zeros((2,), dtype=np.float64)
    chi = np.zeros((2,), dtype=np.complex128)

    nh = nhel * nsf

    if fmass != 0.0:
        pp = min(p[0], np.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2))

        if pp == 0.0:
            sqm[0] = np.sqrt(abs(fmass))  # possibility of negative fermion masses
            sqm[1] = sign(sqm[0], fmass)  # possibility of negative fermion masses
            ip = (1 + nh) / 2
            im = (1 - nh) / 2
            fi[0] = ip * sqm[ip]
            fi[1] = im * nsf * sqm[ip]
            fi[2] = ip * nsf * sqm[im]
            fi[3] = im * sqm[im]
        else:
            sf[0] = float(1 + nsf + (1 - nsf) * nh) * 0.5
            sf[1] = float(1 + nsf - (1 - nsf) * nh) * 0.5
            omega[0] = np.sqrt(p[0] + pp)
            omega[1] = fmass / omega[0]
            ip = (3 + nh) / 2
            im = (3 - nh) / 2
            sfomeg[0] = sf[0] * omega[ip]
            sfomeg[1] = sf[1] * omega[im]
            pp3 = max(pp + p[3], 0.0)
            chi[0] = np.sqrt(pp3 * 0.5 / pp)
            if pp3 == 0.0:
                chi[1] = -nh
            else:
                chi[1] = (nh * p[1] + 1j * p[2]) / np.sqrt(2.0 * pp * pp3)
            fi[0] = sfomeg[0] * chi[im]
            fi[1] = sfomeg[0] * chi[ip]
            fi[2] = sfomeg[1] * chi[im]
            fi[3] = sfomeg[1] * chi[ip]
    else:
        if p[1] == 0 and p[2] == 0 and p[3] < 0:
            sqp0p3 = 0
        else:
            sqp0p3 = np.sqrt(max(p[0] + p[3], 0.0)) * nsf

        chi[0] = sqp0p3
        if sqp0p3 == 0.0:
            chi[1] = (-nhel) * np.sqrt(2.0 * p[0])
        else:
            chi[1] = (nh * p[1] + 1j * p[2]) / sqp0p3

        if nh == 1:
            fi[0] = 0.0
            fi[1] = 0.0
            fi[2] = chi[0]
            fi[3] = chi[1]
        else:
            fi[0] = chi[1]
            fi[1] = chi[0]
            fi[2] = 0.0
            fi[3] = 0.0

    return


#
# double complex fo[5],chi[1]
# double precision p(0:3),sf[1],sfomeg[1],omega[1],fmass,pp,pp3,sqp0p3,sqm(0:1)
# integer nhel,nsf,nh,ip,im
# double precision 0.0, 0.5, 2.0
# parameter( 0.0 = 0.0, 0.5 = 0.5, 2.0 = 2.0 )
def oxxxxx(p, fmass, nhel, nsf, fo):
    fo[4] = (p[0] + 1j * p[3]) * nsf
    fo[5] = (p[1] + 1j * p[2]) * nsf
    nh = nhel * nsf
    if fmass != 0.0:
        pp = min(p[0], np.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2))
        if pp == 0.0:
            sqm[0] = np.sqrt(abs(fmass))  # possibility of negative fermion masses
            sqm[1] = sign(sqm[0], fmass)  # possibility of negative fermion masses
            ip = -((1 + nh) / 2)
            im = (1 - nh) / 2
            fo[0] = im * sqm(im)
            fo[1] = ip * nsf * sqm(im)
            fo[2] = im * nsf * sqm(-ip)
            fo[3] = ip * sqm(-ip)
        else:
            pp = min(p[0], np.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2))
            sf[0] = dble(1 + nsf + (1 - nsf) * nh) * 0.5
            sf[1] = dble(1 + nsf - (1 - nsf) * nh) * 0.5
            omega[0] = np.sqrt(p[0] + pp)
            omega[1] = fmass / omega[0]
            ip = (3 + nh) / 2
            im = (3 - nh) / 2
            sfomeg[0] = sf[0] * omega[ip]
            sfomeg[1] = sf[1] * omega[im]
            pp3 = max(pp + p[3], 0.0)
            chi[0] = np.sqrt(pp3 * 0.5 / pp)
            if pp3 == 0.0:
                chi[1] = -nh
            else:
                chi[1] = (nh * p[1] + 1j * -p[2]) / np.sqrt(2.0 * pp * pp3)
            fo[0] = sfomeg[1] * chi[im]
            fo[1] = sfomeg[1] * chi[ip]
            fo[2] = sfomeg[0] * chi[im]
            fo[3] = sfomeg[0] * chi[ip]
    else:
        if p[1] == 0 and p[2] == 0 and p[3] < 0:
            sqp0p3 = 0
        else:
            sqp0p3 = np.sqrt(max(p[0] + p[3], 0.0)) * nsf
        chi[0] = sqp0p3
        if sqp0p3 == 0.0:
            chi[1] = (-nhel) * np.sqrt(2.0 * p[0])
        else:
            chi[1] = (nh * p[1] + 1j * -p[2]) / sqp0p3
        if nh == 1:
            fo[0] = chi[0]
            fo[1] = chi[1]
            fo[2] = 0.0
            fo[3] = 0.0
        else:
            fo[0] = 0.0
            fo[1] = 0.0
            fo[2] = chi[1]
            fo[3] = chi[0]
    return


#
# This subroutine computes a VECTOR wavefunction.
#
# input:
#       real    p(0:3)         : four-momentum of vector boson
#       real    vmass          : mass          of vector boson
#       integer nhel = -1, 0, 1: helicity      of vector boson
#                                (0 is forbidden if vmass=0.0)
#       integer nsv  = -1 or 1 : +1 for final, -1 for initial
#
# output:
#       complex vc[5]          : vector wavefunction       epsilon^mu(v)
#
# double complex vc[5]
# double precision p(0:3),vmass,hel,hel0,pt,pt2,pp,pzpt,emp,sqh
# integer nhel,nsv,nsvahl

# double precision 0.0, 0.5, 1.0, 2.0
# parameter( 0.0 = 0.0, 0.5 = 0.5 )
# parameter( 1.0 = 1.0, 2.0 = 2.0 )
def vxxxxx(p, vmass, nhel, nsv, vc):
    sqh = np.sqrt(0.5)
    hel = float(nhel)
    nsvahl = nsv * abs(hel)
    pt2 = p[1] ** 2 + p[2] ** 2
    pp = min(p[0], np.sqrt(pt2 + p[3] ** 2))
    pt = min(pp, np.sqrt(pt2))
    vc[4] = (p[0] + 1j * p[3]) * nsv
    vc[5] = (p[1] + 1j * p[2]) * nsv
    if vmass != 0.0:
        hel0 = 1.0 - dabs(hel)
        if pp == 0.0:
            vc[0] = complex(0.0)
            vc[1] = complex(-hel * sqh)
            vc[2] = complex(0.0 + 1j * nsvahl * sqh)
            vc[3] = complex(hel0)
        else:
            emp = p[0] / (vmass * pp)
            vc[0] = complex(hel0 * pp / vmass)
            vc[3] = complex(hel0 * p[3] * emp + hel * pt / pp * sqh)
            if pt != 0.0:
                pzpt = p[3] / (pp * pt) * sqh * hel
                vc[1] = complex(
                    hel0 * p[1] * emp - p[1] * pzpt,
                    -nsvahl * p[2] / pt * sqh,
                )
                vc[2] = complex(
                    hel0 * p[2] * emp - p[2] * pzpt,
                    nsvahl * p[1] / pt * sqh,
                )
            else:
                vc[1] = complex(-hel * sqh)
                vc[2] = complex(0.0, nsvahl * sign(sqh, p[3]))
    else:
        pp = p[0]
        pt = np.sqrt(p[1] ** 2 + p[2] ** 2)
        vc[0] = complex(0.0)
        vc[3] = complex(hel * pt / pp * sqh)
        if pt != 0.0:
            pzpt = p[3] / (pp * pt) * sqh * hel
            vc[1] = complex(-p[1] * pzpt, -nsv * p[2] / pt * sqh)
            vc[2] = complex(-p[2] * pzpt, nsv * p[1] / pt * sqh)
        else:
            vc[1] = complex(-hel * sqh)
            vc[2] = complex(0.0, nsv * sign(sqh, p[3]))
    return


#
# This subroutine computes a complex SCALAR wavefunction.
#
# input:
#       real    p(0:3)         : four-momentum of scalar boson
#       integer nss  = -1 or 1 : +1 for final, -1 for initial
#
# output:
#       complex sc[2]          : scalar wavefunction                   s
#
# double complex sc[2]
# double precision p(0:3)
# integer nss

# double precision 1.0
# parameter( 1.0 = 1.0 )
def sxxxxx(p, nss, sc):
    sc[0] = complex(1.0)
    sc[1] = complex(p[0], p[3]) * nss
    sc[2] = complex(p[1], p[2]) * nss
    return


#
# This subroutine computes an amplitude of the fermion-fermion-vector
# coupling.
#
# input:
#       complex fi[5]          : flow-in  fermion                   |fi>
#       complex fo[5]          : flow-out fermion                   <fo|
#       complex vc[5]          : input    vector                      v
#       complex gc[1]          : coupling constants                  gvf
#
# output:
#       complex vertex         : amplitude                     <fo|v|fi>
#
# double complex fi[5],fo[5],gc[1],vc[5],vertex

# double precision 0.0, 1.0
# parameter( 0.0 = 0.0 )
# double complex cImag, cZero
# parameter( cImag = ( 0.0, 1.0 ), cZero = ( 0.0, 0.0 ) )
def iovxxx(fi, fo, vc, gc, vertex):
    vertex = gc[0] * (
        (fo[2] * fi[0] + fo[3] * fi[1]) * vc[0]
        + (fo[2] * fi[1] + fo[3] * fi[0]) * vc[1]
        - (fo[2] * fi[1] - fo[3] * fi[0]) * vc[2] * cImag
        + (fo[2] * fi[0] - fo[3] * fi[1]) * vc[3]
    )
    if gc[1] != cZero:
        vertex = vertex + gc[1] * (
            (fo[0] * fi[2] + fo[1] * fi[3]) * vc[0]
            - (fo[0] * fi[3] + fo[1] * fi[2]) * vc[1]
            + (fo[0] * fi[3] - fo[1] * fi[2]) * vc[2] * cImag
            - (fo[0] * fi[2] - fo[1] * fi[3]) * vc[3]
        )
    return


#
# This subroutine computes the sum of photon and Z currents with the
# suitable weights ( j(W3) = cos(theta_W) j(Z) + sin(theta_W) j(A) ).
# The output j3 is useful as an input of vvvxxx, jvvxxx or w3w3xx.
# The photon propagator is given in Feynman gauge, and the Z propagator
# is given in unitary gauge.
#
# input:
#       complex fi[5]          : flow-in  fermion                   |fi>
#       complex fo[5]          : flow-out fermion                   <fo|
#       complex gaf[1]         : fi couplings with A                 gaf
#       complex gzf[1]         : fi couplings with Z                 gzf
#       real    zmass          : mass  of Z
#       real    zwidth         : width of Z
#
# output:
#       complex j3[5]          : W3 current             j^mu(<fo|w3|fi>)
#
# double complex fi[5],fo[5],j3[5],gaf[1],gzf[1]
# double complex c0l,c1l,c2l,c3l,csl,c0r,c1r,c2r,c3r,csr,dz,ddif
# double complex gn,gz3l,ga3l
# double complex cm2  ! mass**2- I Gamma mass (Fabio)
# double precision q(0:3),zmass,zwidth,zm2,zmw
# double precision q2,da,ww,cw,sw

# double precision 0.0, 1.0
# parameter( 0.0 = 0.0, 1.0 = 1.0 )
# double complex cImag, cZero
# parameter( cImag = ( 0.0, 1.0 ), cZero = ( 0.0, 0.0 ) )
def j3xxxx(fi, fo, gaf, gzf, zmass, zwidth, j3):
    j3[4] = fo[4] - fi[4]
    j3[5] = fo[5] - fi[5]
    q[0] = -dble(j3[4])
    q[1] = -dble(j3[5])
    q[2] = -dimag(j3[5])
    q[3] = -dimag(j3[4])
    q2 = q[0] ** 2 - (q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    zm2 = zmass**2
    zmw = zmass * zwidth
    da = 1.0 / q2
    # ww = max(dsign(zmw,q2), 0.0)
    dz = 1.0 / (q2 - zm2 + 1j * zmw)
    ddif = (-zm2 + 1j * zmw) * da * dz

    # ddif is the difference : ddif=da-dz
    #  For the running width, use below instead of the above ww,dz and ddif.
    #      ww = max( zwidth*q2/zmass, 0.0 )
    #      dz = 1.0/dcmplx( q2-zm2, zmw )
    #      ddif = dcmplx( -zm2, zmw )*da*dz
    cw = 1.0 / sqrt(1.0 + (gzf[1] / gaf[1]) ** 2)
    sw = sqrt((1.0 - cw) * (1.0 + cw))
    gn = gaf[1] * sw
    gz3l = gzf[0] * cw
    ga3l = gaf[0] * sw
    c0l = fo[2] * fi[0] + fo[3] * fi[1]
    c0r = fo[0] * fi[2] + fo[1] * fi[3]
    c1l = -(fo[2] * fi[1] + fo[3] * fi[0])
    c1r = fo[0] * fi[3] + fo[1] * fi[2]
    c2l = (fo[2] * fi[1] - fo[3] * fi[0]) * cImag
    c2r = (-fo[0] * fi[3] + fo[1] * fi[2]) * cImag
    c3l = -fo[2] * fi[0] + fo[3] * fi[1]
    c3r = fo[0] * fi[2] - fo[1] * fi[3]
    # Fabio's implementation of the fixed width
    cm2 = zm2 + 1j * -zmw
    #     csl = (q[0+1]*c0l-q[1+1]*c1l-q[2+1]*c2l-q[3+1]*c3l)/zm2
    #     csr = (q[0+1]*c0r-q[1+1]*c1r-q[2+1]*c2r-q[3+1]*c3r)/zm2
    csl = (q[0] * c0l - q[1] * c1l - q[2] * c2l - q[3] * c3l) / cm2
    csr = (q[0] * c0r - q[1] * c1r - q[2] * c2r - q[3] * c3r) / cm2

    j3[0] = (
        gz3l * dz * (c0l - csl * q[0])
        + ga3l * c0l * da
        + gn * (c0r * ddif + csr * q[0] * dz)
    )
    j3[1] = (
        gz3l * dz * (c1l - csl * q[1])
        + ga3l * c1l * da
        + gn * (c1r * ddif + csr * q[1] * dz)
    )
    j3[2] = (
        gz3l * dz * (c2l - csl * q[2])
        + ga3l * c2l * da
        + gn * (c2r * ddif + csr * q[2] * dz)
    )
    j3[3] = (
        gz3l * dz * (c3l - csl * q[3])
        + ga3l * c3l * da
        + gn * (c3r * ddif + csr * q[3] * dz)
    )
    return


#
# This subroutine computes an off-shell fermion wavefunction from a
# flowing-OUT external fermion and a vector boson.
#
#
# input:
#       complex fo[5]          : flow-out fermion                   <fo|
#       complex vc[5]          : input    vector                      v
#       complex gc[1]          : coupling constants                  gvf
#       real    fmass          : mass  of OUTPUT fermion f'
#       real    fwidth         : width of OUTPUT fermion f'
#
# output:
#       complex fvo[5]         : off-shell fermion             <fo,v,f'|
#
# double complex fo[5],vc[5],gc[1],fvo[5],sl1,sl2,sr1,sr2,d
# double precision pf(0:3),fmass,fwidth,pf2

# double precision 0.0, 1.0
# parameter( 0.0 = 0.0, 1.0 = 1.0 )
# double complex cImag, cZero
# parameter( cImag = ( 0.0, 1.0 ), cZero = ( 0.0, 0.0 ) )
def fvoxxx(fo, vc, gc, fmass, fwidth, fvo):
    fvo[4] = fo[4] + vc[4]
    fvo[5] = fo[5] + vc[5]
    pf[0] = dble(fvo[4])
    pf[1] = dble(fvo[5])
    pf[2] = dimag(fvo[5])
    pf[3] = dimag(fvo[4])
    pf2 = pf[0] ** 2 - (pf[1] ** 2 + pf[2] ** 2 + pf[3] ** 2)
    d = -1.0 / (pf2 - fmass**2 + 1j * fmass * fwidth)
    sl1 = (vc[0] + vc[3]) * fo[2] + (vc[1] + cImag * vc[2]) * fo[3]
    sl2 = (vc[1] - cImag * vc[2]) * fo[2] + (vc[0] - vc[3]) * fo[3]
    if gc[1] != cZero:
        sr1 = (vc[0] - vc[3]) * fo[0] - (vc[1] + cImag * vc[2]) * fo[1]
        sr2 = -(vc[1] - cImag * vc[2]) * fo[0] + (vc[0] + vc[3]) * fo[1]
        fvo[0] = (
            gc[1] * ((pf[0] + pf[3]) * sr1 + fvo[5] * sr2) + gc[0] * fmass * sl1
        ) * d
        fvo[1] = (
            gc[1] * (dconjg(fvo[5]) * sr1 + (pf[0] - pf[3]) * sr2) + gc[0] * fmass * sl2
        ) * d
        fvo[2] = (
            gc[0] * ((pf[0] - pf[3]) * sl1 - fvo[5] * sl2) + gc[1] * fmass * sr1
        ) * d
        fvo[3] = (
            gc[0] * (-dconjg(fvo[5]) * sl1 + (pf[0] + pf[3]) * sl2)
            + gc[1] * fmass * sr2
        ) * d
    else:
        d = d * gc[0]
        fvo[0] = fmass * sl1 * d
        fvo[1] = fmass * sl2 * d
        fvo[2] = ((pf[0] - pf[3]) * sl1 - fvo[5] * sl2) * d
        fvo[3] = (-dconjg(fvo[5]) * sl1 + (pf[0] + pf[3]) * sl2) * d

    return


#
# This subroutine computes an off-shell fermion wavefunction from a
# flowing-IN external fermion and a vector boson.
#
# input:
#       complex fi[5]          : flow-in  fermion                   |fi>
#       complex vc[5]          : input    vector                      v
#       complex gc[1]          : coupling constants                  gvf
#       real    fmass          : mass  of output fermion f'
#       real    fwidth         : width of output fermion f'
#
# output:
#       complex fvi[5]         : off-shell fermion             |f',v,fi>
#
# double complex fi[5],vc[5],gc[1],fvi[5],sl1,sl2,sr1,sr2,d
# double precision pf(0:3),fmass,fwidth,pf2
#
# double precision 0.0, 1.0
# parameter( 0.0 = 0.0, 1.0 = 1.0 )
# double complex cImag, cZero
# parameter( cImag = ( 0.0, 1.0 ), cZero = ( 0.0, 0.0 ) )
def fvixxx(fi, vc, gc, fmass, fwidth, fvi):
    fvi[4] = fi[4] - vc[4]
    fvi[5] = fi[5] - vc[5]
    pf[0] = dble(fvi[4])
    pf[1] = dble(fvi[5])
    pf[2] = dimag(fvi[5])
    pf[3] = dimag(fvi[4])
    pf2 = pf[0] ** 2 - (pf[1] ** 2 + pf[2] ** 2 + pf[3] ** 2)
    d = -1.0 / dcmplx(pf2 - fmass**2 + 1j * fmass * fwidth)
    sl1 = (vc[0] + vc[3]) * fi[0] + (vc[1] - cImag * vc[2]) * fi[1]
    sl2 = (vc[1] + cImag * vc[2]) * fi[0] + (vc[0] - vc[3]) * fi[1]

    if gc[1] != cZero:
        sr1 = (vc[0] - vc[3]) * fi[2] - (vc[1] - cImag * vc[2]) * fi[3]
        sr2 = -(vc[1] + cImag * vc[2]) * fi[2] + (vc[0] + vc[3]) * fi[3]
        fvi[0] = (
            gc[0] * ((pf[0] - pf[3]) * sl1 - dconjg(fvi[5]) * sl2) + gc[1] * fmass * sr1
        ) * d
        fvi[1] = (
            gc[0] * (-fvi[5] * sl1 + (pf[0] + pf[3]) * sl2) + gc[1] * fmass * sr2
        ) * d
        fvi[2] = (
            gc[1] * ((pf[0] + pf[3]) * sr1 + dconjg(fvi[5]) * sr2) + gc[0] * fmass * sl1
        ) * d
        fvi[3] = (
            gc[1] * (fvi[5] * sr1 + (pf[0] - pf[3]) * sr2) + gc[0] * fmass * sl2
        ) * d
    else:
        d = d * gc[0]
        fvi[0] = ((pf[0] - pf[3]) * sl1 - dconjg(fvi[5]) * sl2) * d
        fvi[1] = (-fvi[5] * sl1 + (pf[0] + pf[3]) * sl2) * d
        fvi[2] = fmass * sl1 * d
        fvi[3] = fmass * sl2 * d
    return


#
# This subroutine computes an off-shell fermion wavefunction from a
# flowing-IN external fermion and a vector boson.
#
# input:
#       complex fi[5]          : flow-in  fermion                   |fi>
#       complex sc[2]          : input    scalar                      s
#       complex gc[1]          : coupling constants                 gchf
#       real    fmass          : mass  of OUTPUT fermion f'
#       real    fwidth         : width of OUTPUT fermion f'
#
# output:
#       complex fsi[5]         : off-shell fermion             |f',s,fi>
#
# double complex fi[5],sc[2],fsi[5],gc[1],sl1,sl2,sr1,sr2,ds
# double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
def fsixxx(fi, sc, gc, fmass, fwidth, fsi):
    fsi[4] = fi[4] - sc[1]
    fsi[5] = fi[5] - sc[2]
    pf[0] = dble(fsi[4])
    pf[1] = dble(fsi[5])
    pf[2] = dimag(fsi[5])
    pf[3] = dimag(fsi[4])
    pf2 = pf[0] ** 2 - (pf[1] ** 2 + pf[2] ** 2 + pf[3] ** 2)
    ds = -sc[0] / (pf2 - fmass**2 + 1j * fmass * fwidth)
    p0p3 = pf[0] + pf[3]
    p0m3 = pf[0] - pf[3]
    sl1 = gc[0] * (p0p3 * fi[0] + dconjg(fsi[5]) * fi[1])
    sl2 = gc[0] * (p0m3 * fi[1] + fsi[5] * fi[0])
    sr1 = gc[1] * (p0m3 * fi[2] - dconjg(fsi[5]) * fi[3])
    sr2 = gc[1] * (p0p3 * fi[3] - fsi[5] * fi[2])
    fsi[0] = (gc[0] * fmass * fi[0] + sr1) * ds
    fsi[1] = (gc[0] * fmass * fi[1] + sr2) * ds
    fsi[2] = (gc[1] * fmass * fi[2] + sl1) * ds
    fsi[3] = (gc[1] * fmass * fi[3] + sl2) * ds
    return


#
# This subroutine computes an off-shell fermion wavefunction from a
# flowing-OUT external fermion and a vector boson.
#
# input:
#       complex fo[5]          : flow-out fermion                   <fo|
#       complex sc[5]          : input    scalar                      s
#       complex gc[1]          : coupling constants                 gchf
#       real    fmass          : mass  of OUTPUT fermion f'
#       real    fwidth         : width of OUTPUT fermion f'
#
# output:
#       complex fso[5]         : off-shell fermion             <fo,s,f'|
#

# double complex fo[5],sc[5],fso[5],gc[1],sl1,sl2,sr1,sr2,ds
# double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
def fsoxxx(fo, sc, gc, fmass, fwidth, fso):
    fso[4] = fo[4] + sc[1]
    fso[5] = fo[5] + sc[2]
    pf[0] = dble(fso[4])
    pf[1] = dble(fso[5])
    pf[2] = dimag(fso[5])
    pf[3] = dimag(fso[4])
    pf2 = pf[0] ** 2 - (pf[1] ** 2 + pf[2] ** 2 + pf[3] ** 2)
    ds = -sc[0] / dcmplx(pf2 - fmass**2, fmass * fwidth)
    p0p3 = pf[0] + pf[3]
    p0m3 = pf[0] - pf[3]
    sl1 = gc[1] * (p0p3 * fo[2] + fso[5] * fo[3])
    sl2 = gc[1] * (p0m3 * fo[3] + dconjg(fso[5]) * fo[2])
    sr1 = gc[0] * (p0m3 * fo[0] - fso[5] * fo[1])
    sr2 = gc[0] * (p0p3 * fo[1] - dconjg(fso[5]) * fo[0])
    fso[0] = (gc[0] * fmass * fo[0] + sl1) * ds
    fso[1] = (gc[0] * fmass * fo[1] + sl2) * ds
    fso[2] = (gc[1] * fmass * fo[2] + sr1) * ds
    fso[3] = (gc[1] * fmass * fo[3] + sr2) * ds
    return


#
# This subroutine computes an amplitude of the fermion-fermion-scalar
# coupling.
#
# input:
#       complex fi[5]          : flow-in  fermion                   |fi>
#       complex fo[5]          : flow-out fermion                   <fo|
#       complex sc[2]          : input    scalar                      s
#       complex gc[1]          : coupling constants                 gchf
#
# output:
#       complex vertex         : amplitude                     <fo|s|fi>
#
# double complex fi[5],fo[5],sc[2],gc[1],vertex
# INTEGER DIM
# PARAMETER(DIM=18)
def iosxxx(fi, fo, sc, gc, vertex):
    vertex = sc[0] * (
        gc[0] * (fi[0] * fo[0] + fi[1] * fo[1])
        + gc[1] * (fi[2] * fo[2] + fi[3] * fo[3])
    )
    return
