import multiprocessing as mp
from data_loader import DataLoader
from spectrum_matcher import SpectrumMatcher
from scoring_function import optimize_q_wrapper
from simple_scoring_function import simple_scoring_function
from visualization import SpectrumVisualizer
from utils import calculate_fdr

def main():
    loader = DataLoader()
    # create the dirs
    loader.create_directories()

    fasta_df = loader.load_fasta("data/uniprotkb_human_proteins_isoforms.fasta")
    
    # Load or generate peptide database
    peptide_df = loader.load_or_generate_peptide_df(fasta_df, "data/peptide_df.csv")

    mgf_files = [
        "data/new_CTR03_BA46_INSOLUBLE_01.mgf",
        "data/new_CTR08_BA46_INSOLUBLE_01.mgf",
        "data/new_CTR45_BA46_INSOLUBLE_01.mgf",
    ]
    mgf_df = loader.load_mgf_files(mgf_files, "data/mgf_df.csv")

    matcher = SpectrumMatcher(peptide_df)

    # Run analysis
    scoring_functions = [optimize_q_wrapper, simple_scoring_function]
    result_df = matcher.match_spectra(mgf_df, scoring_functions)
    
    # Save results
    result_df.to_csv("results/matched_spectra_df.csv")


    try:
        # Calculate FDR and filter results
        fdr_df = calculate_fdr(result_df)
        fdr_df.to_csv("results/fdr_filtered_df.csv")
    except ValueError:
        print("No results to filter")
    except Exception as e:
        print(f"Error filtering results: {e}")

    # Create visualizations
    visualizer = SpectrumVisualizer(result_df)
    
    kde_plot = visualizer.plot_score_distributions()
    kde_plot.savefig("results/score_distributions_kde.png")

    violin_plot = visualizer.plot_violin_distributions()
    violin_plot.write_html("results/score_distributions_violin.html")

    mass_score_plot = visualizer.plot_mass_score_scatter()
    mass_score_plot.savefig("results/mass_score_scatter.png")

    hist_plot = visualizer.plot_score_histogram()
    hist_plot.savefig("results/score_histogram.png")

    confident_matches = visualizer.get_confident_matches(threshold=0.5)
    print(f"{len(confident_matches)} spectra identified with score > treshold")
    confident_matches.to_csv("results/confident_matches.csv")

if __name__ == "__main__":
    main()