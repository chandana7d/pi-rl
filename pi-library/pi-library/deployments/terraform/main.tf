provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "pi_library" {
  ami           = "ami-0c55b159cbfafe1f0" # example AMI
  instance_type = "t2.micro"

  tags = {
    Name = "PiLibraryInstance"
  }
}
