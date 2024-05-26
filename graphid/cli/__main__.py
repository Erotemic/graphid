from scriptconfig.modal import ModalCLI


def main():
    import graphid

    class GraphidCLI(ModalCLI):
        """
        The GraphID CLI
        """
        from graphid.cli.finish_install import FinishInstallCLI

    cli = GraphidCLI()
    cli.version = graphid.__version__
    cli.main(strict=True)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/graphid/graphid/cli/__main__.py
    """
    main()
