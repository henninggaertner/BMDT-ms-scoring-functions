import multiprocessing as mp
from data_loader import DataLoader
from spectrum_matcher import SpectrumMatcher
from utils import calculate_fdr
from scoring_function import optimize_q_wrapper
from simple_scoring_function import simple_scoring_function
from visualization import SpectrumVisualizer

def main():
    loader = DataLoader()

    fasta_df = loader.load_fasta("data/uniprotkb_human_proteins_isoforms.fasta")
    
    # Load or generate peptide database
    peptide_df = loader.load_or_generate_peptide_df(fasta_df, "data/peptide_df.csv")

    # TODO: Load MGF files
    mgf_files = [
        "new_CTR03_BA46_INSOLUBLE_01.mgf",
        "new_CTR08_BA46_INSOLUBLE_01.mgf",
        "new_CTR45_BA46_INSOLUBLE_01.mgf"
    ]
    mgf_df = loader.load_mgf_files(mgf_files, "data/mgf_df.csv")

    matcher = SpectrumMatcher(peptide_df)

    # Run analysis
    scoring_functions = [optimize_q_wrapper, simple_scoring_function] # Andromeda and simple scoring from lecture
    n_processes = 15
    
    result_df = matcher.match_spectra_parallel(
        mgf_df,
        scoring_functions,
        n_processes
    )
    
    # Calculate FDR
    try:
        result_df = calculate_fdr(result_df)
    except Exception as e:
        print(f"Error while calculating FDR: {e}")

    # Save results
    result_df.to_csv("results/matched_spectra_df.csv")

    # Create visualizations
    visualizer = SpectrumVisualizer(result_df)
    
    kde_plot = visualizer.plot_score_distributions()
    kde_plot.savefig("results/score_distributions_kde.png")
    kde_plot.close()

    violin_plot = visualizer.plot_violin_distributions()
    violin_plot.write_html("results/score_distributions_violin.html")

    mass_score_plot = visualizer.plot_mass_score_scatter()
    mass_score_plot.savefig("results/mass_score_scatter.png")
    mass_score_plot.close()

    hist_plot = visualizer.plot_score_histogram()
    hist_plot.savefig("results/score_histogram.png")
    hist_plot.close()

    confident_matches = visualizer.get_confident_matches(threshold=0.5)
    print(f"{len(confident_matches)} spectra identified with score > treshold")
    confident_matches.to_csv("results/confident_matches.csv")

if __name__ == "__main__":
    main()