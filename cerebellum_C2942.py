#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on March 1 2019

@author: Marie Claire Capolei - macca@elektro.dtu.dk
"""

import sys
sys.path.append("/usr/local/lib")
sys.path.append("/usr/local/lib/python2.7/dist-packages")
import numpy as np
from random import *
from lwpr import *

class MCC:
    
    
    def __init__(self,n_uml, n_ccm, n_input_mf):
        # each uml unit learning machine refers to a controlled object uml = [uml_0, ...,uml_i,...,uml_N]
        # each uml contains a number of specialized ccm cerebellar canonical circuit uml_i = [ccm_0,...,ccm_i,...ccm_M]
        # each ccm is specialized on a specific feature (ex position,velocity...)

        self.name = "Cerebellum"
        
        self.n_uml      = n_uml # number of unite learning machine
        self.n_ccm      = n_ccm
        self.n_input_mf = n_input_mf
        self.n_out_pf   = self.n_uml


        self.debug_update     = 0
        self.debug_prediction = 0
        self.rfs_print        = 1

        self.signal_normalization = 1   # normalization of the signal propagating inside
        self.IO_on = True               # include the IO contribution to the DCN
        self.PC_on = True               # include the PC contribution to the DCN
        self.MF_on = True               # include the MF contribution to the DCN


        # *** Teaching Signal ***
        # these values are used to normalize, they need to be initialized in the experiment
        # [uml_0,...,uml_N] = [[ccm_00,...,ccm_0M],...,[ccm_N0,...,ccm_NM]]

        # Normalization input range
        self.range_symmetry = [1, 1]
        self.range_IO       = [[[0.,np.pi] for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.range_signal   = [[0., np.pi] for i in range(0, self.n_uml)]


        # *** Granular Layer Learning Parameters ***

        self.init_D_gr        = 0.5
        self.init_alpha_gr    = 100.
        self.w_gen_gr         = 0.4
        self.diag_only_gr     = bool(1)
        self.update_D_gr      = bool(0)
        self.meta_gr          = bool(0)
        self.init_lambda_gr   = 0.9
        self.tau_lambda_gr    = 0.5
        self.final_lambda_gr  = 0.995
        self.w_prune_gr       = 0.95
        self.meta_rate_gr     = 0.3
        self.add_threshold_gr = 0.95
        self.kernel_gr        = 'Gaussian'
        self.w_gr             = []

        self.x_PF = [0. for k in range(self.n_out_pf)]

        # *** Synaptic weights ***
        
        # PF-PC synaptic weights
        self.delta_w_pf_pc = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.w_pf_pc       = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # PF-DCN synaptic weights
        self.delta_w_pc_dcn = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.w_pc_dcn       = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # IO-DCN synaptic weights
        self.delta_w_io_dcn = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.w_io_dcn       = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # MF-DCN synaptic weights
        self.delta_w_mf_dcn = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.w_mf_dcn       = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # *** Plasticity ***
        #exc
        self.ltp_PF_PC_max  = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.ltd_PF_PC_max  = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        #inh
        self.ltp_PC_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.ltd_PC_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        #exc
        self.ltp_MF_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.ltd_MF_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        #exc
        self.mtp_IO_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.mtd_IO_DCN_max = [[1. * 10**(-3.) for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
             

        self.alphaPF_PC  = [[1. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.alphaPC_DCN = [[1. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.alphaMF_DCN = [[1. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.alphaIO_DCN = [[1. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]


        # *** Initialize the signals ***

        # *** Mossy Fiber ***
        self.x_MF       = [0.  for i in range(0, self.n_input_mf)]
        self.x_MF_teach = [0.  for i in range(0, self.n_uml)]
        self.x_MF_DCN   = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # *** Purkinje Cells ***
        self.x_PC       = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.x_PC_norm  = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.x_PC_DCN   = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # *** Inferior Olive ***
        self.x_IO     = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]
        self.x_IO_DCN = [[0. for j in range(0, self.n_ccm)] for i in range(0, self.n_uml)]

        # *** Deep Cerebellar Nuclei ***
        self.x_DCN = [0. for i in range(0, self.n_uml)]


    def create_model(self):
        print("\n ** Creating Granular Layer **")
        self.model            = LWPR( self.n_input_mf, self.n_out_pf)
        self.model.init_D     = self.init_D_gr*np.eye(self.n_input_mf)
        self.model.init_alpha = self.init_alpha_gr*np.ones([self.n_input_mf, self.n_input_mf])
        self.model.w_gen      = self.w_gen_gr
        self.model.diag_only  = self.diag_only_gr
        self.model.update_D   = self.update_D_gr
        self.model.meta       = self.meta_gr
        #self.model.init_lambda   = init_lambda_gr
        #self.model.tau_lambda    = tau_lambda_gr
        #self.model.final_lambda  = final_lambda_gr
        self.model.w_prune       = self.w_prune_gr
        #self.model.meta_rate     = meta_rate_gr
        #self.model.add_threshold = add_threshold_gr
        #self.model.kernel        = kernel_gr

    def saturate_signal(self, x, range):
        if x > range[1]:
            return range[1]
        elif x < range[0]:
            return range[0]
        else:
            return x

    def norma(self, x, input_range, symmetry):
        if symmetry == 1:
            # Normalization formula : y = (x - min_x)*(max_y - min_y)/(max_x - min_x) + min_y
            zero_in = input_range[1] - sum(map(abs, input_range)) / 2.

            if x >= zero_in:
                min_in = zero_in
                max_in = input_range[1]
                min_out = 0.
                max_out = 1.
            else:
                min_in = input_range[0]
                max_in = zero_in
                max_out = 0.
                min_out = -1.
            return (self.saturate_signal(x, input_range) - min_in) * (max_out - min_out) / (max_in - min_in) + min_out
        else:
            return (self.saturate_signal(x, input_range) - input_range[0]) * (1. - 0.) / (input_range[1] - input_range[0]) + 0.

    def nl_activation(self,x):
        return (2./ (1. + np.exp(-2.*x)) )-1.

    def dbg(self, signal, uml, ccm, x):
        print("\n %s %s[%i][%i] = " %(self.name, signal, uml,ccm) + str(x))

    def update_model(self):

        self.model.update(np.array([n for n in self.x_MF]), np.array([teach for teach in self.x_MF_teach]))
        if self.debug_update == 1:
            print("\n "+self.name+" -- Updating model"+"\n sensory input: "+str( np.array([ n for n in self.x_MF])  )+"\n teaching signal"+str(np.array([ teach for teach in self.x_MF_teach ])))

        if self.rfs_print == 1:
            print("\n "+self.name+" -- rfs :"+str(self.model.num_rfs))


    def prediction(self, error):

        # ----------------------------------------------------------------------------#
        # ----------------------- ***  Parallel Fiber prediction *** -----------------#
        # ----------------------------------------------------------------------------#
        
        (self.x_PF, self.w_gr) = self.model.predict(np.array([n for n in self.x_MF]))
        
        if self.debug_prediction == 1:
            self.dbg("output_PF", 0, 0, self.x_PF)
            self.dbg("w_gr", 0, 0, self.w_gr)

        for uml in range(0,self.n_uml):
            for ccm in range(0, self.n_ccm):
                # ----------------------------------------------------------------------------#
                # ---------- *** Parallel Fibers - Purkinje cells with IO modulation *** -----------#
                # ----------------------------------------------------------------------------#


                # ------------------ *** Inferior Olive - parallel fiber *** -----------------------#
                if self.signal_normalization == 1:
                    self.x_IO[uml][ccm] = self.norma(error[ccm][uml], self.range_IO[uml][ccm], self.range_symmetry[0])
                    if self.debug_prediction == 1:
                        self.dbg("input_IO", uml, ccm, self.x_IO[uml][ccm])
                else:
                    self.x_IO[uml][ccm] = error[ccm][uml]

                # ------------------ *** Plasticity Pf-Pc *** -----------------------#
                # delta_w = ltp/(|io|+1)^alpha - ltd*|io|
                self.delta_w_pf_pc[uml][ccm]  = self.ltp_PF_PC_max[uml][ccm]/( abs(self.x_IO[uml][ccm]) + 1.)**(self.alphaPF_PC[uml][ccm]) - self.ltd_PF_PC_max[uml][ccm]*abs(self.x_IO[uml][ccm])
                self.w_pf_pc[uml][ccm] += self.delta_w_pf_pc[uml][ccm]

                # ------------------ *** output signal Purkinje Cell *** -----------------------#
                self.x_PC[uml][ccm] = self.w_pf_pc[uml][ccm]*self.x_PF[uml]

                if self.debug_prediction == 1:
                    self.dbg("w_pf_pc", uml, ccm, self.w_pf_pc[uml][ccm])
                    self.dbg("output_PC", uml, ccm, self.x_PC[uml][ccm])

                # ----------------------------------------------------------------------------#
                # -------------- *** Purkinje - Deep Cerebellar Nuclei *** -------------------#
                # ----------------------------------------------------------------------------#

                if self.signal_normalization == 1:
                    self.x_PC_norm[uml][ccm] = self.norma(self.x_PC[uml][ccm], self.range_signal[uml],self.range_symmetry[1])
                else:
                    self.x_PC_norm[uml][ccm] = self.x_PC[uml][ccm]

                # ----------------------- *** Plasticity Pc-DCN *** -------------------------# --> modulated by dcn(t-1) and pc
                # luque 2014
                # delta_w = (ltp*|pc|)^alpha * (1 - 1/(1 + |dcn|)^alpha) - ltd*(1- |pc|)
                self.delta_w_pc_dcn[uml][ccm] = ( self.ltp_PC_DCN_max[uml][ccm]*(abs(self.x_PC_norm[uml][ccm])**self.alphaPC_DCN[uml][ccm]) )*(1.-1./( 1. + abs(self.norma( self.x_DCN[uml], self.range_signal[uml],self.range_symmetry[1])))**self.alphaPC_DCN[uml][ccm] ) - self.ltd_PC_DCN_max[uml][ccm]*( 1.- abs(self.x_PC_norm[uml][ccm]))
                self.w_pc_dcn[uml][ccm] += self.delta_w_pc_dcn[uml][ccm]

                # ------------- *** Input signal from Purkinje Cell to DCN ***----------------#
                self.x_PC_DCN[uml][ccm] = self.x_PC_norm[uml][ccm]*self.w_pc_dcn[uml][ccm]

                if self.debug_prediction == 1:
                    self.dbg("w_pc_dcn", uml, ccm, self.w_pc_dcn[uml][ccm])
                    self.dbg("output_PC_DCN", uml, ccm, self.x_PC_DCN[uml][ccm])

                # ----------------------------------------------------------------------------#
                # ------------- *** Mossy Fibers - Deep Cerebellar Nuclei *** ----------------#
                # ----------------------------------------------------------------------------#

                # ------------------------ *** Plasticity MF-DCN *** -------------------------#
                # delta_w = ltp/(|pc|+1)^alpha - ltd*|pc|
                self.delta_w_mf_dcn[uml][ccm]  = self.ltp_MF_DCN_max[uml][ccm]/( abs(self.x_PC_norm[uml][ccm]) + 1.)**self.alphaMF_DCN[uml][ccm] - self.ltd_MF_DCN_max[uml][ccm]*abs(self.x_PC_norm[uml][ccm])
                self.w_mf_dcn[uml][ccm] += self.delta_w_mf_dcn[uml][ccm]

                # --------------------- *** Input signal from MF to DCN ***--------------------#
                self.x_MF_DCN[uml][ccm] = self.w_mf_dcn[uml][ccm]*self.x_MF_teach[uml]

                if self.debug_prediction == 1:
                    self.dbg("w_mf_dcn", uml, ccm, self.w_mf_dcn[uml][ccm])
                    self.dbg("output_MF_DCN", uml, ccm, self.x_MF_DCN[uml][ccm])


                # ----------------------------------------------------------------------------#
                # ------------- *** Inferior Olive - Deep Cerebellar Nuclei *** --------------#
                # ----------------------------------------------------------------------------#

                # ------------------------ *** Plasticity IO-DCN *** -------------------------#
                # delta_w = ltp*|io| - ltd/(|io|+1)^alpha
                self.delta_w_io_dcn[uml][ccm]  = self.mtp_IO_DCN_max[uml][ccm]*abs(self.x_IO[uml][ccm]) - self.mtd_IO_DCN_max[uml][ccm]/( (abs(self.x_IO[uml][ccm]) + 1.)**self.alphaIO_DCN[uml][ccm] )
                self.w_io_dcn[uml][ccm] += self.delta_w_io_dcn[uml][ccm]

                # --------------------- *** Input signal from IO to DCN ***--------------------#
                self.x_IO_DCN[uml][ccm] =  self.w_io_dcn[uml][ccm]*self.x_IO[uml][ccm]

                if self.debug_prediction == 1:
                    self.dbg("w_io_dcn", uml, ccm, self.w_io_dcn[uml][ccm])
                    self.dbg("output_IO_DCN", uml, ccm, self.x_IO_DCN[uml][ccm])
            
            # ----------------------------------------------------------------------------#
            # ----------------------- *** Deep Cerebellar Nuclei *** ---------------------#
            # ----------------------------------------------------------------------------#            
            # DCN = - (PC_pos + PC_vel) + (MF_pos + MF_vel) + (IO_pos + IO_vel)
            self.x_DCN[uml] =  self.nl_activation( [ 0. , -self.nl_activation(sum(self.x_PC_DCN[uml][:]) ) ][self.PC_on == True]   + [ 0., self.nl_activation(sum(self.x_MF_DCN[uml][:]) ) ] [ self.MF_on == True ] + [ 0. ,self.nl_activation(sum(self.x_IO_DCN[uml][:]))  ][ self.IO_on == True ] )

            if self.debug_prediction == 1:
                self.dbg("output_DCN", uml, 0, self.x_DCN[uml])

        return self.x_DCN
