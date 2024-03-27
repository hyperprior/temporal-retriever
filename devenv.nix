{ pkgs, ... }:

{
 
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = with pkgs; [ 
    git 
    mise
    poetry
    httpie
    jq
    jqp
  ];

  languages.python.enable = true;
  languages.python.version = "3.10.8";

  starship = {
    enable = true;
    config.enable = true;
  };

  pre-commit.hooks = {
    shellcheck.enable = true;
    mdsh.enable = true;
    black.enable = true;
  };

  scripts.menu.exec = "git --version";

  enterShell = ''
    menu
  '';

}
