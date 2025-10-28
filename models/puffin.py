# models/puffin.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_fftconv import FFTConv1d
from torch.autograd import grad


class Puffin(nn.Module):
    """
    Puffin Model for interpretable transcription initiation prediction.
    Single-target output architecture for simplified usage.
    """

    def __init__(self, config):
        super(Puffin, self).__init__()

        self.config = config
        model_config = config['model']

        # Sequence pattern detection layers
        self.conv_motif = nn.Conv1d(4, model_config['num_motifs'],
                                    kernel_size=model_config['motif_kernel_size'],
                                    padding=model_config['motif_kernel_size'] // 2)

        self.conv_initiator = nn.Conv1d(4, model_config['num_initiators'] // 2,
                                        kernel_size=model_config['initiator_kernel_size'],
                                        padding=model_config['initiator_kernel_size'] // 2)

        self.conv_trinucleotide = nn.Conv1d(4, model_config['num_trinucleotides'] // 2,
                                            kernel_size=model_config['trinucleotide_kernel_size'],
                                            padding=model_config['trinucleotide_kernel_size'] // 2)

        # Activation functions
        self.activation = nn.Softplus()
        self.softplus = nn.Softplus()

        # Effect application layers - single target output
        self.deconv_motif = FFTConv1d(model_config['num_motifs'] * 2, 1,
                                      kernel_size=model_config['motif_effect_range'],
                                      padding=model_config['motif_effect_range'] // 2)

        self.deconv_initiator = nn.ConvTranspose1d(model_config['num_initiators'], 1,
                                                   kernel_size=model_config['initiator_effect_range'],
                                                   padding=model_config['initiator_effect_range'] // 2)

        self.deconv_trinucleotide = FFTConv1d(model_config['num_trinucleotides'], 1,
                                              kernel_size=model_config['trinucleotide_effect_range'],
                                              padding=model_config['trinucleotide_effect_range'] // 2)

        # Learnable parameters
        self.scaler = nn.Parameter(torch.ones(1))
        self.scaler2 = nn.Parameter(torch.ones(1))

        self.use_cuda = model_config['use_cuda']
        if self.use_cuda:
            self.cuda()

        # Important: Use generic motif names to clearly indicate limitations
        self.motifnames_original = [f"Learned_Motif_{i:02d}" for i in range(model_config['num_motifs'] * 2)]

        print("Note: Motif names are generic identifiers for demonstration.")
        print("Real biological motif discovery requires multi-stage training and validation.")

    def forward(self, x):
        """Forward pass - single target output"""
        # Compute activations for both strands
        y_motif = torch.cat([self.conv_motif(x), self.conv_motif(x.flip([1, 2])).flip([2])], 1)
        y_initiator = torch.cat([self.conv_initiator(x), self.conv_initiator(x.flip([1, 2])).flip([2])], 1)
        y_trinucleotide = torch.cat([self.conv_trinucleotide(x), self.conv_trinucleotide(x.flip([1, 2])).flip([2])], 1)

        # Apply activation functions
        y_motif_act = self.activation(y_motif)
        y_initiator_act = self.activation(y_initiator)
        y_trinucleotide_act = self.activation(y_trinucleotide)

        # Compute effects and combine - single target output
        motif_effect = self.deconv_motif(y_motif_act)
        initiator_effect = self.deconv_initiator(y_initiator_act)
        trinucleotide_effect = self.deconv_trinucleotide(y_trinucleotide_act)

        # Combine all effects
        y_pred = self.softplus(motif_effect + initiator_effect + trinucleotide_effect)

        return y_pred

    def get_motif_activations(self, seq_tensor):
        """Get motif activation scores"""
        with torch.no_grad():
            # Compute motif activations
            preact_motif = torch.cat([self.conv_motif(seq_tensor),
                                      self.conv_motif(seq_tensor.flip([1, 2])).flip([2])], 1)
            postact_motif = self.activation(preact_motif)

            # Convert to numpy and trim boundaries
            motif_activations = postact_motif.cpu().numpy()[0, :, 325:-325]

        return motif_activations

    def get_motif_effects(self, seq_tensor):
        """Get motif effects - fixed for single target dimension"""
        with torch.no_grad():
            # Compute motif activations
            preact_motif = torch.cat([self.conv_motif(seq_tensor),
                                      self.conv_motif(seq_tensor.flip([1, 2])).flip([2])], 1)
            postact_motif = self.activation(preact_motif)

            # Compute effects for each motif
            dweight = self.deconv_motif.weight.cpu().detach().numpy()
            motif_effects = {}

            for i, motif_name in enumerate(self.motifnames_original):
                # Use convolution to compute motif effects
                effect = np.convolve(
                    postact_motif[0, i, :].cpu().numpy(),
                    dweight[0, i, ::-1],  # Single target, index [0, i, :]
                    mode="same"
                )[325:-325]
                motif_effects[motif_name] = effect

        return motif_effects

    def basepair_contribution_transcription_initiation(self, seq_tensor):
        """Basepair contribution analysis for transcription initiation"""
        original_training = self.training
        self.eval()

        try:
            seq_len = seq_tensor.shape[2]
            trim_size = 325
            output_len = seq_len - 2 * trim_size

            seq_tensor = seq_tensor.clone().detach()
            seq_tensor.requires_grad = True

            # Compute activations for three components
            preact_motif = torch.cat([self.conv_motif(seq_tensor),
                                      self.conv_motif(seq_tensor.flip([1, 2])).flip([2])], 1)
            preact_inr = torch.cat([self.conv_initiator(seq_tensor),
                                    self.conv_initiator(seq_tensor.flip([1, 2])).flip([2])], 1)
            preact_sim = torch.cat([self.conv_trinucleotide(seq_tensor),
                                    self.conv_trinucleotide(seq_tensor.flip([1, 2])).flip([2])], 1)

            postact_motif = self.activation(preact_motif)
            postact_inr = self.activation(preact_inr)
            postact_sim = self.activation(preact_sim)

            # Detach variables for gradient computation
            postact_motif_detached = postact_motif.detach()
            postact_motif_detached.requires_grad = True
            postact_inr_detached = postact_inr.detach()
            postact_inr_detached.requires_grad = True
            postact_sim_detached = postact_sim.detach()
            postact_sim_detached.requires_grad = True

            # Compute prediction
            pred = self.softplus(
                self.deconv_motif(postact_motif_detached) +
                self.deconv_initiator(postact_inr_detached) +
                self.deconv_trinucleotide(postact_sim_detached)
            )

            # Numerical stability fix
            pred_output = pred[:, 0, trim_size:trim_size + output_len]
            log_base = np.log(10)
            exponent = pred_output / log_base
            predexp = torch.exp(exponent * log_base) - 1

            # Handle large values
            if torch.isinf(predexp).any():
                large_mask = torch.isinf(predexp)
                predexp[large_mask] = pred_output[large_mask] * 1000

            predexp_sum = predexp.sum()
            predexp_sum.backward(retain_graph=True)

            # Recompute sequence gradients
            self.zero_grad()
            seq_tensor_grad = seq_tensor.detach()
            seq_tensor_grad.requires_grad = True

            # Recompute activations
            preact_motif_grad = torch.cat([self.conv_motif(seq_tensor_grad),
                                           self.conv_motif(seq_tensor_grad.flip([1, 2])).flip([2])], 1)
            postact_motif_grad = self.activation(preact_motif_grad)

            preact_inr_grad = torch.cat([self.conv_initiator(seq_tensor_grad),
                                         self.conv_initiator(seq_tensor_grad.flip([1, 2])).flip([2])], 1)
            postact_inr_grad = self.activation(preact_inr_grad)

            preact_sim_grad = torch.cat([self.conv_trinucleotide(seq_tensor_grad),
                                         self.conv_trinucleotide(seq_tensor_grad.flip([1, 2])).flip([2])], 1)
            postact_sim_grad = self.activation(preact_sim_grad)

            # Compute combined gradient contribution
            total_grad_effect = (
                    (postact_motif_grad ** 2 * postact_motif_detached.grad).sum() +
                    (postact_inr_grad ** 2 * postact_inr_detached.grad).sum() +
                    (postact_sim_grad ** 2 * postact_sim_detached.grad).sum()
            )

            total_grad_effect.backward()

            # Compute total contribution
            seq_contribution = (
                    (seq_tensor_grad.grad * seq_tensor_grad.data).sum(axis=1).cpu().detach().numpy() -
                    (seq_tensor_grad.grad * (1 - seq_tensor_grad.data)).sum(axis=1).cpu().detach().numpy() / 3
            )

            # Compute per-motif contributions
            tss_contr = {}

            for i, motif_name in enumerate(self.motifnames_original):
                self.zero_grad()
                seq_tensor_motif = seq_tensor.detach()
                seq_tensor_motif.requires_grad = True

                preact_motif_motif = torch.cat([self.conv_motif(seq_tensor_motif),
                                                self.conv_motif(seq_tensor_motif.flip([1, 2])).flip([2])], 1)
                postact_motif_motif = self.activation(preact_motif_motif)

                # Gradient computation
                (postact_motif_motif[:, i, :] ** 2 * postact_motif_detached.grad[:, i, :]).sum().backward()

                # Contribution calculation
                motif_contribution = (
                        (seq_tensor_motif.grad * seq_tensor_motif.data).sum(axis=1).cpu().detach().numpy() -
                        (seq_tensor_motif.grad * (1 - seq_tensor_motif.data)).sum(axis=1).cpu().detach().numpy() / 3
                )

                # Trim boundaries
                if motif_contribution.shape[1] >= trim_size + output_len:
                    tss_contr[motif_name] = motif_contribution[0, trim_size:trim_size + output_len]
                else:
                    tss_contr[motif_name] = motif_contribution[0, :output_len]

            # Return total contribution (trimmed)
            if seq_contribution.shape[1] >= trim_size + output_len:
                final_contribution = seq_contribution[0, trim_size:trim_size + output_len]
            else:
                final_contribution = seq_contribution[0, :output_len]

            return tss_contr, final_contribution

        except Exception as e:
            print(f"Error in contribution analysis: {e}")
            import traceback
            traceback.print_exc()

            # Return zero arrays with correct shape
            output_len = seq_len - 2 * trim_size if 'seq_len' in locals() and 'trim_size' in locals() else 150
            tss_contr = {motif: np.zeros(output_len) for motif in self.motifnames_original}
            return tss_contr, np.zeros(output_len)

        finally:
            if original_training:
                self.train()

    def basepair_contribution_motif(self, seq_tensor):
        """Analyze basepair contributions to motif activation"""
        batch_size = len(self.motifnames_original)

        # Create batch data
        seqts = []
        for j in range(batch_size):
            seqts.append(seq_tensor)
        seq_batch = torch.cat(seqts, axis=0)
        seq_batch.requires_grad = True

        # Compute motif activations
        preact_motif = torch.cat([self.conv_motif(seq_batch),
                                  self.conv_motif(seq_batch.flip([1, 2])).flip([2])], 1)
        postact_motif = self.activation(preact_motif)

        # Create diagonal matrix to select each motif
        if self.use_cuda:
            eye_matrix = torch.eye(batch_size).cuda()[:, :, None]
        else:
            eye_matrix = torch.eye(batch_size)[:, :, None]

        # Compute gradients
        ((postact_motif * eye_matrix) ** 2).sum().backward()

        # Compute contributions
        motifact_seq = (
                (seq_batch.grad * seq_batch.data).sum(axis=1).cpu().detach().numpy() -
                (seq_batch.grad * (1 - seq_batch.data)).sum(axis=1).cpu().detach().numpy() / 3
        )[..., 325:-325]  # Trim boundaries

        motif_contr = {}
        for motif_n, motif_name in enumerate(self.motifnames_original):
            motif_contr[motif_name] = motifact_seq[motif_n, :]

        return motif_contr

    def interpret(self, seq_bp):
        """Complete interpretability analysis"""
        # Sequence encoding
        seq = self.sequence_to_onehot(seq_bp)
        seq_bp_trimmed = seq_bp[325:-325] if len(seq_bp) > 650 else seq_bp

        # Prepare input tensor
        if self.use_cuda:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).cuda()
        else:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            pred0 = self(seq_tensor)
            pred0_np = pred0.cpu().numpy()[0, 0, 325:-325]  # Single target output

        # Get motif activations and effects
        motif_activations = self.get_motif_activations(seq_tensor)
        motif_effects = self.get_motif_effects(seq_tensor)

        # Compute contributions
        print("Calculating transcription initiation contributions...")
        tss_contr, total_contribution = self.basepair_contribution_transcription_initiation(seq_tensor)
        print("Calculating motif activation contributions...")
        motif_contr = self.basepair_contribution_motif(seq_tensor)

        # Build result DataFrame
        lines = {}
        lines["Coordinate"] = list(range(len(seq_bp_trimmed)))
        lines["Sequence"] = list(seq_bp_trimmed)
        lines["Prediction"] = pred0_np

        # Add motif activations
        for i, motif in enumerate(self.motifnames_original):
            if i < len(motif_activations):
                lines[motif + " motif activation"] = motif_activations[i]

        # Add motif effects
        for motif in self.motifnames_original:
            if motif in motif_effects:
                lines[motif + " motif effect"] = motif_effects[motif]

        # Ensure motif transcription initiation contributions are correctly generated
        bp_score_list = []
        for motif in self.motifnames_original:
            if motif in tss_contr:
                contribution_data = tss_contr[motif]
                if isinstance(contribution_data, np.ndarray) and len(contribution_data) > 0:
                    if len(contribution_data) == len(seq_bp_trimmed):
                        bp_score_list.append(contribution_data)
                        lines[motif + " Basepair contribution score to transcription initiation"] = contribution_data
                    else:
                        # Handle length mismatch
                        if len(contribution_data) < len(seq_bp_trimmed):
                            padded = np.pad(contribution_data, (0, len(seq_bp_trimmed) - len(contribution_data)),
                                            'constant')
                            lines[motif + " Basepair contribution score to transcription initiation"] = padded
                        else:
                            lines[
                                motif + " Basepair contribution score to transcription initiation"] = contribution_data[
                                :len(seq_bp_trimmed)]
                else:
                    # Create zero array if data is invalid
                    lines[motif + " Basepair contribution score to transcription initiation"] = np.zeros(
                        len(seq_bp_trimmed))

        # Total transcription initiation contribution
        if len(bp_score_list) > 0:
            total_bp_score = np.sum(np.array(bp_score_list), axis=0)
            lines["Basepair contribution score to transcription initiation"] = total_bp_score
        else:
            # Use total contribution if no motif contributions
            if isinstance(total_contribution, np.ndarray) and len(total_contribution) == len(seq_bp_trimmed):
                lines["Basepair contribution score to transcription initiation"] = total_contribution
            else:
                lines["Basepair contribution score to transcription initiation"] = np.zeros(len(seq_bp_trimmed))

        # Add motif activation contributions
        for motif in self.motifnames_original:
            if motif in motif_contr:
                activation_data = motif_contr[motif]
                if isinstance(activation_data, np.ndarray) and len(activation_data) == len(seq_bp_trimmed):
                    lines[motif + " Basepair contribution score to motif activation"] = activation_data
                else:
                    # Handle length mismatch
                    if len(activation_data) < len(seq_bp_trimmed):
                        padded = np.pad(activation_data, (0, len(seq_bp_trimmed) - len(activation_data)), 'constant')
                        lines[motif + " Basepair contribution score to motif activation"] = padded
                    else:
                        lines[motif + " Basepair contribution score to motif activation"] = activation_data[
                            :len(seq_bp_trimmed)]

        df = pd.DataFrame.from_dict(lines, orient="index")

        print(
            f"Generated data for {len([k for k in lines.keys() if 'Basepair contribution score to transcription initiation' in k])} motifs")

        # Add important notes
        print("\n" + "=" * 80)
        print("INTERPRETATION RESULTS - IMPORTANT NOTES:")
        print("=" * 80)
        print("1. Motif names (Learned_Motif_00, etc.) are generic identifiers.")
        print("2. These do NOT necessarily correspond to real biological motifs.")
        print("3. Real applications require multi-stage training and motif validation.")
        print("4. Current analysis demonstrates methodology but biological interpretation")
        print("   should be done with caution and proper validation.")
        print("=" * 80 + "\n")

        return df

    def sequence_to_onehot(self, sequence):
        """Convert sequence to one-hot encoding"""
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        onehot = np.zeros((4, len(sequence)), dtype=np.float32)

        for i, base in enumerate(sequence.upper()):
            if base in base_to_index:
                onehot[base_to_index[base], i] = 1.0

        return onehot

    def save_model(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)

    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']