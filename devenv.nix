{ pkgs, ... }:

{
 
  # https://devenv.sh/basics/
  env.GREET = "devenv";
  env.LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  

  # https://devenv.sh/packages/
  packages = with pkgs; [ 
    git 
    mise
    poetry
    httpie
    jq
    ruff
    jqp
    uv
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
