import torch
import igv_notebook
import pandas as pd
import pyBigWig
import pandas as pd
import pyfaidx
import pandas as pd
import numpy as np
import kipoiseq
import subprocess
from enformer_lucidrains_pytorch.enformer_pytorch import Enformer
from kipoiseq import Interval
from tqdm import tqdm
from IPython.display import HTML, display

# Slight modification has been made to port it to Torch instead of tensorflow


# GLOBAL VARs
SEQUENCE_LENGTH = 196608

# Functions 
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

class EnformerModel:
    def __init__(self, model_path="EleutherAI/enformer-official-rough"):
        if torch.cuda.is_available():
            print("Using NVIDIA GPU")
            device = torch.device("cuda")
        else:
            print("Using CPU")
            device = torch.device("cpu")

        self.device = device
        self.model = Enformer.from_pretrained(model_path).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
    
    
    
class SEQ_EXTRACT:
    def __init__(self, data, endo_label=False):
        self.data = pd.read_csv(data, sep='\t')
        self.endo = endo_label
        # deleting the index column
        if self.data.index.name is not None:
            del self.data['Unnamed: 0']

    def extract_seq (self, tag, cell_type=None):
        # NOTE: good way of extracting out data
        if not self.endo:
            return self.data.query(f'TAG == "{tag}" and CELL_TYPE == "{cell_type}" ').copy()
        else:
            return self.data.query('TAG.str.contains(@tag)').copy()
    def __repr__(self):
        display(self.data.groupby(['TAG', 'CELL_TYPE']).count())
        return  'Data structure'
    

class EnformerOps:
    def __init__(self):
        self.df_sizes = pd.read_table('../data/hg38.chrom.sizes', header=None).head(22)
        self.tracks = []
        self.input_sequences_file_path = None
        self.interval_list = None
        self.capture_bigwig_names = None
        self.full_generated_range_start = None
        self.full_generated_range_end = None
        self.loaded_seqs = None
        self.model = EnformerModel()
        self.fasta_extractor = FastaStringExtractor("../data/genome.fa")


    def add_track(self, track):
        """
        Adds a track to the list of tracks to be visualized.

        Args:
            track (dict): A dictionary specifying the track to be added.
            Should have the keys "name", "file", "color", "type", and "id"
            (if type is "enformer").
        """
        self.tracks.append(track)


    def load_data(self, input_sequences_file_path):

        if type(input_sequences_file_path) == list:
            self.loaded_seqs =  [ [x] for x in input_sequences_file_path]
            self.input_sequences_file_path = input_sequences_file_path

    def generate_plot_number(self,
                             sequence_number_thousand,
                             step=-1,
                             interval_list=None,
                             show_track=True,
                             capture_bigwig_names=True,
                             wildtype=False):
        """
        Generates IGV tracks for a given sequence in a diffusion dataset.

        Args:
            sequence_number_thousand (int): The number of the sequence ID in
            the diffusion sequences FASTA dataset.

            step (int, optional): Which diffusion step to use. Default is -1,
            which means the last diffusion step (i.e., the final diffused sequence).

            interval_list (list, optional): Coordinate to insert the 200 bp
            sequence. Should be in BED format (chr, start, end). Default is None.

            show_track (bool, optional): Whether to generate IGV tracks as a result.
            Default is True.

            capture_bigwig_names (bool, optional): Whether to output a list with
            all IGV tracks generated and used (in case real bigwig files were used)
            for the final visualization. Default is True.

            wildtype (bool, False)
            Dont insert and capture the wildtype sequence
        Returns:
            list: A list with the name of all bigwig files generated.
        """
        capture_bigwig_names = [] # return the name of all bigwig
        USE_INTERVAL = interval_list
        if not interval_list:
            USE_INTERVAL = self.interval_list # this should be your 200 bp region

        if USE_INTERVAL is None:
            raise ValueError("Interval list must be specified.")

        target_interval = kipoiseq.Interval(USE_INTERVAL[0], USE_INTERVAL[1], USE_INTERVAL[2])

        chr_test = target_interval.resize(SEQUENCE_LENGTH).chr
        start_test = target_interval.resize(SEQUENCE_LENGTH).start
        end_test = target_interval.resize(SEQUENCE_LENGTH).end

        seq_to_mod = self.fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))

        all_seqs_test = self.loaded_seqs[sequence_number_thousand]

        SEQ_IN = self.insert_seq(all_seqs_test[step], seq_to_mod, dont_insert=wildtype) # JUST THE LAST

        sequence_one_hot = one_hot_encode(SEQ_IN)
        
        # NOTE: there is a change here ZELUN
        with torch.no_grad():
            predictions = self.model.forward(torch.from_numpy(sequence_one_hot[np.newaxis]))['human'][0]

        mod_start = int(start_test + ((end_test - start_test)/2)) - int(114688/2)
        mod_end = int(start_test + ((end_test - start_test)/2)) + int(114688/2)


        self.full_generated_range_start = mod_start
        self.full_generated_range_end = mod_end
        self.full_generated_chr = chr_test
            
        b = None

        if show_track:
            igv_notebook.init()
            b = igv_notebook.Browser({
                "genome": "hg38",
                "locus": f"{chr_test}:{mod_start}-{mod_end}"
            })

        for track in self.tracks:
            if track['type'] == 'enformer':
                id = track['id']
                n = track['name']
                lg = track['log']

                p_values = predictions[:, id]
                if lg == True:
                    
                    p_values =np.log10(1 + predictions[:, id].cpu().detach().numpy())
                    
                # aa_milne_arr = [x for x in "abcdefghijklmnopqrstuvwxyz1234567890"]
                # random_var = ''.join(np.random.choice(aa_milne_arr, 20))
                random_var = "predicted"
                
                capture_bigwig_names.append(f"{n}_{random_var}.bw")
                # print(f"chr_test: {chr_test} mod_start: {mod_start} p_values: {p_values} n: {n}")
                out_track = self._enformer_bigwig_creation(chr_test, mod_start, p_values, n, random_var=random_var) # change this pretiction t/name for a real thing
                if show_track:
                    b.load_track(out_track)

            elif track['type'] == 'real':
                n = track['name']
                f = track['file']
                c = track['color']
                capture_bigwig_names.append(f)
                if show_track:
                    b.load_track(self._generate_real_tracks(n, f, c))

        self.capture_bigwig_names = capture_bigwig_names
        return b
        # return capture_bigwig_names




    def capture_full_cords(self):
        if self.full_generated_range_start:
            return  self.full_generated_chr , self.full_generated_range_start, self.full_generated_range_end
        else:
            print ('Run generate_plot_number before it')


    def extract_from_position(self, position, as_dataframe=False):
        """
        Extracts data from the bigwig files generated by generate_plot_number for a given genomic region.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the region.
            end (int): The end position of the region.

        Returns:
            list: A list of dictionaries containing the name of each bigwig file
            and the values for the given region.
        """
        if self.capture_bigwig_names is None:
            raise ValueError("Must call generate_plot_number first to generate the bigwig files.")

        results = []

        for name in self.capture_bigwig_names:
            bw = pyBigWig.open(name)
            values = bw.values(position[0], position[1], position[2])
            results.append({
                'name': name,
                'values': values
            })
        if as_dataframe:
          results = pd.DataFrame({ k['name']: k['values'] for k in results })
        return results

    @staticmethod
    def insert_seq(seq_x, seq_mod_in, dont_insert=False):
        '''
        This function inserts a sequence `seq_x` into a larger sequence `seq_mod_in`.

        Args:
            seq_x (str): The sequence to be inserted into `seq_mod_in`.
            seq_mod_in (str): The larger sequence that `seq_x` will be inserted into.
            dont_insert (bool, optional): Whether or not to skip inserting `seq_x`.
            If `True`, `seq_mod_in` will be returned unchanged. Default is `False`.

        Returns:
            str: The modified sequence with `seq_x` inserted into `seq_mod_in`.
        '''
        seq_to_mod_array = np.array(list(seq_mod_in))
        seq_mod_center = seq_to_mod_array.shape[0] // 2
        if not dont_insert:
            seq_to_mod_array[seq_mod_center - 100:seq_mod_center + 100] = np.array(list(seq_x))
        else:
            print('Keeping endogenous sequence...')
        return ''.join(seq_to_mod_array)

    @staticmethod
    def _enformer_bigwig_creation(chr_name, start, values, track_name, color='BLUE', random_var=''):
        """
        Creates a bigwig file for an Enformer track.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the track.
            values (np.array): The values to be used in the track.
            track_name (str): The name of the track.
            color (str, optional): The color to use for the track. Default is 'BLUE'.

        Returns:
            dict: A dictionary containing the name and path of the bigwig file,
            as well as its format, display mode, and color.
        """
        #TODO: change the track_name each time
        # Makeing         mktemp update the 20 number
        t_name = f"{track_name}_{random_var}.bw"
        bw = pyBigWig.open(t_name, "w")
        df_sizes = pd.read_table('../data/hg38.chrom.sizes', header=None).head(22)
        bw.addHeader([(chr_name, coord) for chr_name, coord in df_sizes.values])
        
        # Convert tensor to NumPy array
        if isinstance(values, torch.Tensor):
            values = values.cpu().detach().numpy()

        values_conversion = (values * 1000 ).astype(np.int64) + 0.0
        bw.addEntries(chr_name, [start + (128 * x) for x in range(values_conversion.shape[0])], values=values_conversion, span=128)
        return {
            "name": f"{track_name}",
            "path": f"{track_name}_{random_var}.bw",
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": f"{color}",
            "height": 100,
        }


    def _generate_real_tracks(self, name, filename, color):
        """
        Generates a real track for a given bigwig file.

        Args:
            name (str): The name of the track.
            file (str): The name of the bigwig file to use for the track.
            color (str): The color to use for the track.


        Returns:
            dict: A dictionary containing the name, path, format, display mode, and color of the track.
        """


        chr_name, start, end = self.capture_full_cords()
        t_name = f"{name}_minimal.bw"
        
        # Execute the shell command
        # !echo '' > $t_name

        # Specify the file path

        # Create an empty file
        with open(t_name, "w"):
            pass

        # subprocess.run(["echo", "", ">", t_name], shell=True)

        bw = pyBigWig.open(t_name, "w")
        bw.addHeader([(chr_name, coord) for chr_name, coord in self.df_sizes.values])
        bw_cut = pyBigWig.open(filename, "r")
        values = np.array(bw_cut.values(chr_name, start, end))
        
        values_conversion = (values * 1000 ).astype(np.int64) + 0.0
        print (values_conversion)
        print (chr_name, start)
        bw.addEntries(chr_name, [r for r in  range(start, start+len(values_conversion)) ] , values=list(values_conversion), span=1, step =1)
        bw.close()
        bw_cut.close()


        return {
            "name": name,
            "path": t_name,
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": color,
            "height": 100,
            }

    def tiling(self, interval_to_window, window=2000, slice=200):
            slice_len = int(((interval_to_window[2] + window) - (interval_to_window[1] - window)) / slice)
            start_slice = (interval_to_window[1] - window)
            slices_position = [[interval_to_window[0], start_slice + (slice * n), start_slice + ((slice * n) + slice)] for n in range(slice_len)]
            return slices_position

    def generate_tiling(self, coord_to_tile, gata_gene_region):
        #TODO df_sizes currently hardcoded since I am not sure what this is
        # df_sizes = pd.read_table('hg38.chrom.sizes', header=None).head(22)
        tiling_coords = self.tiling(coord_to_tile, window=2000)
        regions_capture = []

        t_name = "tiling_vis_"+str(coord_to_tile[1])+"_"+str(coord_to_tile[2])+".bw"
        
#         !rm $t_name
#         !echo '' > $t_name
        
        # Remove the file
        subprocess.run(["rm", t_name])
        # Create an empty file
        subprocess.run(["echo", "''", ">", t_name], shell=True)
        
        bw_insert = pyBigWig.open(t_name, "w")
        bw_insert.addHeader([(chr, coord) for chr, coord in self.df_sizes.values])

        for t in tqdm(tiling_coords):
            bw_list = self.generate_plot_number(120, 1, interval_list=t, show_track=False)
            return_bw_by_tile = self.extract_from_position(gata_gene_region)
            mean_values_region_cage = np.mean(return_bw_by_tile[1]['values']).astype(np.int64) + 0.0
            bw_insert.addEntries(t[0], [t[1]], values=[mean_values_region_cage], span=200)
            regions_capture.append(mean_values_region_cage)

        bw_insert.close()
        return regions_capture
