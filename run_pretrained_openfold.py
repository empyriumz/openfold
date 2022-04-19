# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import numpy as np
import os

import random
import sys
import time
import torch

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax
from openfold.utils.import_weights import (
    import_jax_weights_,
)
from openfold.utils.tensor_utils import (
    tensor_tree_map,
)

from scripts.utils import add_data_args


import warnings
warnings.filterwarnings("ignore", message="Ignoring unrecognized record"
                        )
def _file_name(args, model_name, tag, plddt, relaxed=False):
    if args.phenix_pdb_model is not None and args.single_template_recycle is None:
        file_name = "{}_{}_recycle_{:.2f}.pdb".format(tag, model_name, plddt)
    elif args.phenix_pdb_model is None and args.single_template_recycle is not None:
        file_name = "{}_{}_template_{:.2f}.pdb".format(tag, model_name, plddt)
    elif args.phenix_pdb_model is not None and args.single_template_recycle is not None:
        file_name = "{}_{}_recycle_template_{:.2f}.pdb".format(tag, model_name, plddt)
    else:
        file_name = "{}_{}_{:.2f}.pdb".format(tag, model_name, plddt)
    if not relaxed:
        file_name = "unrelaxed_" + file_name

    return file_name


def main(args):
    best_plddt = 0.0
    # model_list = [
    #     "model_1_ptm",
    #     "model_2_ptm",
    #     "model_3_ptm",
    #     "model_4_ptm",
    #     "model_5_ptm",
    # ]
    model_list = ["model_1", "model_2", "model_3", "model_4", "model_5"]
    # model_list = ["model_3_ptm"]
    for model_name in model_list:
        config = model_config(model_name)
        if args.single_template_recycle is not None:
            config.data.common.use_templates = True
            config.model.template.enabled = True
            config.data.predict.max_templates = 1
            config.data.predict.max_template_hits = 1
            config.model.extra_msa.enabled = False
        # else:
        #     # Terwilliger paper disables template
        #     # to avoid AlphaFold finding existing models with similar sequences
        #     config.data.common.use_templates = False
        #     config.model.template.enabled = False
        # config.model.extra_msa.enabled = False
        # config.model.template.embed_angles = False
        # config.data.predict.max_templates = 1
        # config.data.predict.max_template_hits = 1
        if args.phenix_pdb_model is not None:
            print("Enhanced recycling activated; disable template")
            config.data.common.use_templates = False
            config.model.template.enabled = False
        if args.num_recycle:
            config.data.common.max_recycling_iters = args.num_recycle
        model = AlphaFold(config)
        model = model.eval()
        if args.param_path is None:
            args.param_path = os.path.join(
                "openfold", "resources", "params", "params_" + model_name + ".npz"
            )
        import_jax_weights_(model, args.param_path, version=model_name)
        model = model.to(args.model_device)

        if args.single_template_recycle is None:
            template_featurizer = templates.TemplateHitFeaturizer(
                mmcif_dir=args.template_mmcif_dir,
                max_template_date=args.max_template_date,
                max_hits=config.data.predict.max_templates,
                kalign_binary_path=args.kalign_binary_path,
                release_dates_path=args.release_dates_path,
                obsolete_pdbs_path=args.obsolete_pdbs_path,
            )
        else:
            template_featurizer = None

        use_small_bfd = args.bfd_database_path is None

        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        output_dir_base = args.output_dir
        random_seed = args.data_random_seed
        if random_seed is None:
            random_seed = random.randrange(sys.maxsize)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        if not os.path.exists(output_dir_base):
            os.makedirs(output_dir_base)
        if args.use_precomputed_alignments is None:
            alignment_dir = os.path.join(output_dir_base, "alignments")
        else:
            alignment_dir = args.use_precomputed_alignments

        # Gather input sequences
        with open(args.fasta_path, "r") as fp:
            data = fp.read()

        lines = [
            l.replace("\n", "")
            for prot in data.split(">")
            for l in prot.strip().split("\n", 1)
        ][1:]
        tags, seqs = lines[::2], lines[1::2]

        for tag, seq in zip(tags, seqs):
            fasta_path = os.path.join(args.output_dir, "tmp.fasta")
            with open(fasta_path, "w") as fp:
                fp.write(f">{tag}\n{seq}")

            logging.info("Generating features...")
            local_alignment_dir = os.path.join(alignment_dir, tag)
            if args.use_precomputed_alignments is None:
                if not os.path.exists(local_alignment_dir):
                    os.makedirs(local_alignment_dir)

                alignment_runner = data_pipeline.AlignmentRunner(
                    jackhmmer_binary_path=args.jackhmmer_binary_path,
                    hhblits_binary_path=args.hhblits_binary_path,
                    hhsearch_binary_path=args.hhsearch_binary_path,
                    uniref90_database_path=args.uniref90_database_path,
                    mgnify_database_path=args.mgnify_database_path,
                    bfd_database_path=args.bfd_database_path,
                    uniclust30_database_path=args.uniclust30_database_path,
                    pdb70_database_path=args.pdb70_database_path,
                    use_small_bfd=use_small_bfd,
                    no_cpus=args.cpus,
                )
                alignment_runner.run(fasta_path, local_alignment_dir)

            feature_dict = data_processor.process_fasta(
                fasta_path=fasta_path, alignment_dir=local_alignment_dir
            )
            if args.single_template_recycle is not None:
                # feature_dict = templates.single_template_process(
                #     feature_dict, args.single_template_recycle
                # )
                # feature_dict = templates._deprecated_single_template_process(
                #     feature_dict, args.single_template_recycle
                # )
                # import pickle

                # with open("template_feature_7ku7.pkl", "rb") as f:
                #     template_feature = pickle.load(f)

                # feature_dict = {**feature_dict, **template_feature}
                import pickle

                with open("all_feature_7ku7.pkl", "rb") as f:
                    feature_dict = pickle.load(f)
                feature_dict["template_all_atom_mask"] = feature_dict.pop("template_all_atom_masks")

            # import pickle
            # with open("feature_dict_old.pkl", "wb") as f:
            #     pickle.dump(feature_dict, f)
            # Remove temporary FASTA file
            os.remove(fasta_path)

            processed_feature_dict = feature_processor.process_features(
                feature_dict,
                mode="predict",
            )

            if args.phenix_pdb_model is not None:
                """
                Import the pdb model generated by Phenix, then get the coordinates of all C_beta atoms (or C_alpha for glycine)
                which will be used for recycling
                """
                import Bio

                parser = Bio.PDB.PDBParser()
                structure = parser.get_structure("phenix", args.phenix_pdb_model)
                # structure_unrelax = parser.get_structure("unrelaxed", "/host/openfold/unrelaxed_7LCI_model_5_ptm_71.52_rebuilt_with_autosharp.pdb")
                # res_list = Bio.PDB.Selection.unfold_entities(structure, "R")
                # res_list_unrelax = Bio.PDB.Selection.unfold_entities(structure_unrelax, "R")

                # from Bio import pairwise2

                # res_name = []
                # res_name_unrelax = []
                # for res in res_list:
                #     res_name.append(res.resname) 
                # for res in res_list_unrelax:
                #     res_name_unrelax.append(res.resname) 
    
                # alignments = pairwise2.align.localms(res_name,res_name_unrelax,2,-1,-0.5,-0.1, gap_char=["-"])
                # from itertools import cycle

                # res_cycle = cycle(res_list_unrelax)
                # res_unrelax_aligned = []
                # for i, res_name in enumerate(alignments[0].seqB):
                #     if res_name == '-':
                #         res_unrelax_aligned.append(i)
                #     else:
                #         res_unrelax_aligned.append(next(res_cycle))
                
                # coord = []
                # for res_unrelax, res_relax in zip(res_unrelax_aligned, res_list):
                #     res = res_relax if isinstance(res_unrelax, int) else res_unrelax
                #     if res.resname == "GLY":
                #         atom = res["CA"]
                #         coord.append(atom.get_coord())
                #     else:
                #         atom = res["CB"]
                #         coord.append(atom.get_coord())
                res_list = Bio.PDB.Selection.unfold_entities(structure, "R")
                coord = []
                for res in res_list:
                    if res.resname == "GLY":
                        atom = res["CA"]
                        coord.append(atom.get_coord())
                        # if atom.get_bfactor() >= 30:
                        #     coord.append(atom.get_coord())
                        # else:
                        #     coord.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                    else:
                        atom = res["CB"]
                        coord.append(atom.get_coord())
                        # if atom.get_bfactor() >= 30:
                        #     coord.append(atom.get_coord())
                        # else:
                        #     coord.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
                        
                recycle_dim = processed_feature_dict["aatype"].shape[-1]
                processed_feature_dict["x_prev"] = torch.tensor(
                    np.repeat(np.array(coord)[..., None], recycle_dim, axis=-1)
                )

            # with open("processed_feature_normal_old.pkl", "wb") as f:
            #     pickle.dump(processed_feature_dict, f)

            logging.info("Executing model...")
            for key in processed_feature_dict.keys():
                if processed_feature_dict[key].dtype == torch.float64:
                    processed_feature_dict[key] = processed_feature_dict[key].to(
                        torch.float32
                    )
                # if processed_feature_dict[key].dtype == torch.float32:
                #     processed_feature_dict[key] = processed_feature_dict[key].to(torch.float64)
            # model = model.to(torch.float64)
            batch = processed_feature_dict
            with torch.no_grad():
                batch = {
                    k: torch.as_tensor(v, device=args.model_device)
                    for k, v in batch.items()
                }

                t = time.perf_counter()
                out = model(batch)
                logging.info(f"Inference time: {time.perf_counter() - t}")

            # Toss out the recycling dimensions --- we don't need them anymore
            batch = tensor_tree_map(lambda x: np.array(x[..., -1].cpu()), batch)
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            mean_plddt = np.mean(out["plddt"])
            print("Mean plddt for {}: {:.2f}".format(model_name, mean_plddt))
            if mean_plddt > best_plddt:
                best_plddt = mean_plddt
                best_output = [batch, out, model_name]
            if "predicted_tm_score" in out.keys():
                print("Predicted TM score: {:.2f}".format(out["predicted_tm_score"]))

    batch, out, model_name = best_output[0], best_output[1], best_output[2]
    plddt = out["plddt"]
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    unrelaxed_protein = protein.from_prediction(
        features=batch, result=out, b_factors=plddt_b_factors
    )
    # Save the unrelaxed PDB
    file_name = _file_name(args, model_name, tag, best_plddt, relaxed=False)
    unrelaxed_output_path = os.path.join(
        args.output_dir,
        file_name,
    )
    with open(unrelaxed_output_path, "w") as f:
        f.write(protein.to_pdb(unrelaxed_protein))

    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(args.model_device != "cpu"),
        **config.relax,
    )
    if np.mean(plddt) >= 70:
        # Relax the prediction.
        t = time.perf_counter()
        relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

        logging.info(f"Relaxation time: {time.perf_counter() - t}")
        print(f"Relaxation time: {time.perf_counter() - t}")

        file_name = _file_name(args, model_name, tag, best_plddt, relaxed=True)
        # Save the relaxed PDB.
        relaxed_output_path = os.path.join(args.output_dir, file_name)

        with open(relaxed_output_path, "w") as f:
            f.write(relaxed_pdb_str)

    # if args.save_outputs:
    #     output_dict_path = os.path.join(
    #         args.output_dir, f"{tag}_{args.model_name}_output_dict.pkl"
    #     )
    #     with open(output_dict_path, "wb") as fp:
    #         pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_path",
        type=str,
    )
    parser.add_argument(
        "--template_mmcif_dir",
        default=None,
        type=str,
        help="Path to find template files in mmcif format",
    )
    parser.add_argument(
        "--single_template_recycle",
        type=str,
        default=None,
        help="""Using pdb model refined by Phenix as the template.""",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=None,
        help="""# of recycling in the inference.""",
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--phenix_pdb_model",
        type=str,
        default=None,
        help="""Path for the PDB model generated by Phenix""",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda:0",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_1",
        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm, as defined on the AlphaFold GitHub.""",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        default=None,
        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             openfold/resources/params""",
    )
    parser.add_argument(
        "--save_outputs",
        type=bool,
        default=False,
        help="Whether to save all model outputs, including embeddings, etc.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="""Number of CPUs with which to run alignment tools""",
    )
    parser.add_argument(
        "--preset", type=str, default="full_dbs", choices=("reduced_dbs", "full_dbs")
    )
    parser.add_argument("--data_random_seed", type=str, default=None)
    add_data_args(parser)
    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )
    logging.basicConfig(filename="example.log", level=logging.DEBUG)
    main(args)
